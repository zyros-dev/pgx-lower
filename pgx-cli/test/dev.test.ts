import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { RunResult, StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { formatWorkflowSummary, runDevCommand } from "../src/dev.js";
import { gateFailureStatePath, recordGateFailure } from "../src/gate-memory.js";

class FakeRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];
  results: RunResult[] = [];
  streamingResults: Array<{ match: string; exitCode: number; stdout?: string; stderr?: string }> = [];

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    const rendered = [command, ...args].join(" ");
    if (command === "git" && rendered.includes("branch --show-current")) {
      return { exitCode: 0, stdout: "feature-a\n", stderr: "" };
    }
    if (command === "git" && rendered.includes("status --short")) {
      return { exitCode: 0, stdout: "", stderr: "" };
    }
    if (command === "git" && rendered.includes("rev-parse HEAD")) {
      return { exitCode: 0, stdout: "abc123\n", stderr: "" };
    }
    if (command === "ssh" && rendered.includes("git branch --show-current")) {
      return { exitCode: 0, stdout: "feature-a\n", stderr: "" };
    }
    if (command === "ssh" && rendered.includes("git status --short") && !rendered.includes("--branch")) {
      return { exitCode: 0, stdout: "", stderr: "" };
    }
    if (command === "ssh" && rendered.includes("git rev-parse HEAD")) {
      return { exitCode: 0, stdout: "abc123\n", stderr: "" };
    }
    if (command === "mutagen" && args[1] === "list") {
      return { exitCode: 0, stdout: healthyJson(), stderr: "" };
    }
    if (rendered.includes("pg_regress") || rendered.includes("ctest -V")) {
      return { exitCode: 0, stdout: "1: ok 1 - 1_one_tuple 10 ms\n", stderr: "" };
    }
    return this.results.shift() ?? { exitCode: 0, stdout: "", stderr: "" };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args });
    const rendered = [command, ...args].join(" ");
    const scriptedIndex = this.streamingResults.findIndex((result) => rendered.includes(result.match));
    if (scriptedIndex >= 0) {
      const [scripted] = this.streamingResults.splice(scriptedIndex, 1);
      const stdout = scripted.stdout ?? "";
      const stderr = scripted.stderr ?? "";
      options.stdout?.write(stdout);
      options.stderr?.write(stderr);
      return {
        childExitCode: scripted.exitCode,
        stdoutSample: sample(stdout),
        stderrSample: sample(stderr),
        timedOut: false
      };
    }
    const stdout = command === "mutagen" && args[1] === "list"
      ? healthyJson()
      : rendered.includes(".pgx-cli/sync-probes/")
        ? `probe-${rendered.match(new RegExp("sync-probes/([^/']+)\\.txt"))?.[1] ?? "missing"}\n`
        : rendered.includes("pg_regress") || rendered.includes("ctest -V")
          ? "1: ok 1 - 1_one_tuple 10 ms\n"
          : "ok\n";
    options.stdout?.write(stdout);
    return {
      childExitCode: 0,
      stdoutSample: { head: stdout, tail: "", truncated: false },
      stderrSample: { head: "", tail: "", truncated: false },
      timedOut: false
    };
  }
}

function sample(text: string) {
  return { head: text, tail: "", truncated: false };
}

function healthyJson(): string {
  return JSON.stringify([{ name: "pgx-lower", paused: false, status: "watching", alpha: { connected: true }, beta: { connected: true } }]);
}

const devConfigBase = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower",
  dockerContainer: "pgx-lower-dev",
  buildQueue: "pgx-build",
  checkQueue: "pgx-check",
  sync: {
    required_for_remote: true,
    flush_timeout_seconds: 45,
    proof: {
      enabled: false,
      path: ".pgx-cli/sync-probes",
      required_for: ["build", "test", "lint", "psql", "pg_regress", "bench", "profile", "run"]
    }
  },
  output: {
    mode: "agent",
    transcript_dir: ".pgx-cli/runs",
    max_lines_per_step: 20,
    max_lines_total: 80,
    failure_tail_lines: 20,
    success_tail_lines: 10,
    progress: "final-summary",
    full_output_requires_flag: true
  }
};

function makeDevConfig() {
  const root = mkdtempSync(join(tmpdir(), "pgx-dev-test-"));
  mkdirSync(join(root, "pgx-cli/src"), { recursive: true });
  mkdirSync(join(root, "pgx-cli/dist"), { recursive: true });
  mkdirSync(join(root, "src/pgx-lower/test"), { recursive: true });
  mkdirSync(join(root, "tests/unit-tests/sql"), { recursive: true });
  mkdirSync(join(root, ".pgx-cli/runs/install"), { recursive: true });
  writeFileSync(join(root, "pgx-cli/src/index.ts"), "export {};\n");
  writeFileSync(join(root, "pgx-cli/dist/index.js"), "export {};\n");
  writeFileSync(join(root, "src/pgx-lower/test/type_mapping_tests.cpp"), "PGX_TEST_FN(type_mapping_smoke) {}\n");
  writeFileSync(join(root, "tests/unit-tests/sql/type_mapping.sql"), "select 1;\n");
  writeFileSync(join(root, ".pgx-cli/runs/install/summary.json"), JSON.stringify({
    runId: "install",
    commandName: "dev-build-install-debug",
    command: [
      "ssh",
      "comfy",
      "bash",
      "-c",
      "'export PATH=$HOME/.local/bin:$PATH && cd /home/zel/repos/pgx-lower && docker exec pgx-lower-dev cmake --install build-artifacts/ptest'"
    ],
    workflowExitCode: 0,
    finishedAt: "2099-01-01T00:00:00.000Z",
    gitHead: "abc123",
    artifactPaths: ["/workspace/build-artifacts/ptest"]
  }, null, 2) + "\n");
  writeFileSync(join(root, "tests/pg_regress_baseline.txt"), "");
  return { ...devConfigBase, localProjectPath: root };
}

const oldLintScript = ["scripts", "run_lint.sh"].join("/");
const oldBaselineScript = ["ptest", "with", "baseline.py"].join("_");

describe("dev workflow summaries", () => {
  test("formats passing workflow steps", () => {
    expect(
      formatWorkflowSummary([
        { name: "lint diff", command: ["pgx-cli", "dev", "lint", "diff"], exitCode: 0, summary: "clean" },
        { name: "compile", command: ["pgx-cli", "dev", "build", "compile"], exitCode: 0, logPath: "/tmp/pgx-compile.out" }
      ])
    ).toBe(
      [
        "Workflow summary:",
        "- ok lint diff: pgx-cli dev lint diff - clean",
        "- ok compile: pgx-cli dev build compile - log /tmp/pgx-compile.out",
        "Workflow result: ok"
      ].join("\n") + "\n"
    );
  });

  test("formats failing workflow steps", () => {
    expect(
      formatWorkflowSummary([
        { name: "lint diff", command: ["pgx-cli", "dev", "lint", "diff"], exitCode: 1, jobId: "12" }
      ])
    ).toContain("- fail lint diff: pgx-cli dev lint diff - job 12");
  });
});

describe("dev commands", () => {
  test("dev status checks sync, branch, and queues", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["status"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("mutagen sync list pgx-lower");
    expect(commands).toContain("git status --short --branch");
    expect(commands).toContain("TS_SOCKET=/tmp/pgx-build.sock tsp");
    expect(commands).toContain("TS_SOCKET=/tmp/pgx-check.sock tsp");
    expect(output.stdout).toContain("run id:");
    expect(output.stdout).toContain("transcript:");
    expect(output.stdout).toContain("dev status");
  });

  test("dev logs latest tails the build queue", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["logs", "latest"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("TS_SOCKET=/tmp/pgx-build.sock tsp -t");
    expect(output.stdout).toContain("run id:");
  });

  test("dev logs id tails a specific build queue job", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["logs", "7"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("TS_SOCKET=/tmp/pgx-build.sock tsp -t 7");
    expect(output.stdout).toContain("transcript:");
  });

  test("dev preflight dispatches to strict preflight", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["preflight", "--strict"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    expect(output.stdout).toContain("dev preflight: ok");
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("git -C");
    expect(commands).toContain("branch --show-current");
    expect(commands).toContain("status --short");
    expect(commands).toContain("rev-parse HEAD");
    expect(commands).toContain("mutagen sync list pgx-lower");
    expect(commands).toContain(".pgx-cli/sync-probes/dev-preflight.txt");
  });

  test.each([
    [["lint", "diff"], "ssh", "clang-tidy-diff-20"],
    [["check", "diff"], "ssh", "clang-format-diff-20"],
    [["format", "diff"], "ssh", "clang-format-diff-20"],
    [["lint", "file", "src/pgx-lower/runtime/tuple_access.cpp"], "ssh", "clang-tidy-20"],
    [["lint", "files", "a.cpp", "b.cpp"], "ssh", "clang-tidy-20"],
    [["test", "unit", "type_mapping"], "ssh", "type_mapping.sql"],
    [["test", "tpch"], "ssh", "pg_regress"],
    [["test", "focused"], "ssh", "UTEST-PG_OK"]
  ])("dev %s runs direct workflow commands", async (args, command, marker) => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(args, runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(runner.calls.some((call) => call.command === command && call.args.join(" ").includes(marker))).toBe(true);
    expect(output.stdout).toContain("run id:");
    expect(output.stdout).toContain("transcript:");
    expect(runner.calls.some((call) => call.command === "python3")).toBe(false);
    expect(commands).not.toContain(oldLintScript);
    expect(commands).not.toContain(oldBaselineScript);
    expect(commands).not.toContain("just");
  });

  test("dev format diff flushes formatted remote changes back locally", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["format", "diff"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" "));
    const formatIndex = commands.findIndex((command) => command.startsWith("ssh ") && command.includes("clang-format-diff-20"));
    const lastFlushIndex = commands.findLastIndex((command) => command === "mutagen sync flush pgx-lower");
    expect(formatIndex).toBeGreaterThan(-1);
    expect(lastFlushIndex).toBeGreaterThan(formatIndex);
  });


  test("dev lint diff runs directly on thor when invoked from the remote checkout", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = { ...makeDevConfig(), runningOnRemote: true };
    const exitCode = await runDevCommand(["lint", "diff"], runner, output, config);

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("docker exec -i pgx-lower-dev");
    expect(commands).not.toContain("mutagen sync flush");
    expect(runner.calls.some((call) => call.command === "ssh")).toBe(false);
  });


  test("dev gate batch runs directly on thor when invoked from the remote checkout", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = { ...makeDevConfig(), runningOnRemote: true };
    const exitCode = await runDevCommand(["gate", "batch"], runner, output, config);

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("clang-format-diff-20");
    expect(commands).toContain("clang-tidy-diff-20");
    expect(commands).toContain("docker exec -i pgx-lower-dev");
    expect(commands).not.toContain("mutagen sync flush");
    expect(runner.calls.some((call) => call.command === "ssh")).toBe(false);
  });


  test("dev gate review builds pgx-cli before full-lint CMake configure in a clean checkout", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["gate", "review"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("npm --prefix pgx-cli install && npm --prefix pgx-cli run build");
    expect(commands.indexOf("npm --prefix pgx-cli install")).toBeLessThan(commands.indexOf("build-docker-lint"));
  });


  test("dev focused unit tests use generated unit-tests path", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["test", "focused"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("npm --prefix pgx-cli run build");
    expect(commands).toContain("/workspace/pgx-cli/dist/index.js test unit-sql --root /workspace");
    expect(commands).toContain("tests/unit-tests/sql");
    expect(commands).not.toContain("tests/regress-unit/sql");
  });

  test("CTest registrations include route assertions and unit SQL", () => {
    const extensionCmake = readFileSync(new URL("../../extension/CMakeLists.txt", import.meta.url), "utf8");

    expect(extensionCmake).toContain("pgx_lower_regress_routes");
    expect(extensionCmake).toContain("pgx_lower_tpch_routes");
    expect(extensionCmake).toContain("pgx_lower_compare_postgres_tpch");
    expect(extensionCmake).toContain("pgx_lower_regress_unit");
    expect(extensionCmake).toContain("route-check");
    expect(extensionCmake).toContain("compare-postgres-internal");
    expect(extensionCmake).toContain("PGX_COMPARE_POSTGRES_INTERNAL=1");
    expect(extensionCmake).toContain("test-runs/ctest-tpch-correctness/compare-postgres");
    expect(extensionCmake).toContain("tests/unit-tests/sql");
    expect(extensionCmake).toContain("test unit-sql");
    expect(extensionCmake).toContain("pgx-cli/dist/index.js");
    expect(extensionCmake).toContain("build pgx-cli before configuring CMake");
    expect(extensionCmake).not.toContain("find_program(PGX_CLI_EXECUTABLE pgx-cli)");
    expect(extensionCmake).not.toContain("set(PGX_CLI_EXECUTABLE pgx-cli)");
    expect(extensionCmake).not.toContain("npm --prefix");
    expect(extensionCmake).not.toContain("dist/unit-sql.js");
  });

  test("dev gate batch runs diff-scoped checks", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["gate", "batch"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("clang-format-diff-20");
    expect(commands).toContain("clang-tidy-diff-20");
    expect(commands).toContain("UTEST-PG_OK");
    expect(commands).not.toContain("just");
    expect(output.stdout).toContain("Workflow result: ok");
  });

  test("dev gate review runs full review checks", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = makeDevConfig();
    const exitCode = await runDevCommand(["gate", "review"], runner, output, config);

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("clang-format-diff-20");
    expect(commands).toContain("clang-tidy-20");
    expect(commands).not.toContain(oldLintScript);
    expect(output.stdout).toContain("pgx-cli dev build compile --profile debug");
    expect(output.stdout).toContain("transcript:");
    expect(commands).toContain("UTEST-PG_OK");
    expect(commands).toContain("npm --prefix pgx-cli run build");
    expect(commands).not.toContain("npm --prefix /workspace/pgx-cli run build");
    expect(commands).toContain("ctest -V");
    expect(commands).toContain("compare-postgres-internal --workload tpch-correctness");
    expect(commands).toContain("PGX_COMPARE_POSTGRES_INTERNAL=1");
    expect(output.stdout).toContain("pgx-cli test compare-postgres --workload tpch-correctness");
    const waitIndex = commands.indexOf('tsp -w "$id"');
    const catIndex = commands.indexOf('tsp -c "$id"');
    expect(waitIndex).toBeGreaterThan(-1);
    expect(catIndex).toBeGreaterThan(waitIndex);
	    expect(commands).not.toContain(oldBaselineScript);
    expect(commands).not.toContain("just");
    expect(output.stdout).not.toContain("stdout preview:");
	  });

  test("dev gate review persists a full workflow summary for pr readiness", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = makeDevConfig();
    const exitCode = await runDevCommand(["gate", "review"], runner, output, config);

    expect(exitCode).toBe(0);
    const reviewRunId = output.stdout.match(/^run id: (.+dev-gate-review.+)$/m)?.[1];
    expect(reviewRunId).toEqual(expect.any(String));
    const summary = JSON.parse(readFileSync(join(config.localProjectPath, ".pgx-cli", "runs", reviewRunId ?? "", "summary.json"), "utf8"));
    expect(summary).toMatchObject({
      runId: reviewRunId,
      commandName: "dev-gate-review",
      command: ["pgx-cli", "dev", "gate", "review"],
      workflowExitCode: 0,
      gitHead: "abc123"
    });
    expect(summary.steps).toEqual(expect.arrayContaining([
      expect.objectContaining({
        name: "compare-postgres",
        command: ["pgx-cli", "test", "compare-postgres", "--workload", "tpch-correctness"],
        exitCode: 0
      })
    ]));
  });

  test("dev gate review no-bench does not run bench", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["gate", "review", "--no-bench"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    expect(output.stdout).not.toContain("bench");
  });

  test("failed review writes gate state", async () => {
    const runner = new FakeRunner();
    runner.streamingResults.push({ match: "build-docker-lint", exitCode: 1, stderr: "lint failed\n" });
    const output = { stdout: "", stderr: "" };
    const config = makeDevConfig();
    const exitCode = await runDevCommand(["gate", "review"], runner, output, config);

    expect(exitCode).toBe(1);
    const state = JSON.parse(readFileSync(gateFailureStatePath(config.localProjectPath), "utf8"));
    expect(state).toMatchObject({
      gate: "review",
      head: "abc123",
      stepName: "lint",
      focusedCommand: ["pgx-cli", "dev", "lint", "diff"]
    });
    expect(state.runId).toEqual(expect.any(String));
    expect(state.transcript).toEqual(expect.stringContaining(".pgx-cli/runs/"));
  });

  test.each([
    {
      match: "build-docker-lint",
      stepName: "lint",
      focusedCommand: ["pgx-cli", "dev", "lint", "diff"]
    },
    {
      match: "cmake --build .",
      stepName: "compile",
      focusedCommand: ["pgx-cli", "dev", "build", "compile", "--profile", "debug"],
      beforeFailure: [{ match: "build-docker-lint", exitCode: 0 }]
    },
    {
      match: "ctest -V",
      stepName: "test",
      focusedCommand: ["pgx-cli", "dev", "test", "focused"]
    },
    {
      match: "compare-postgres-internal --workload tpch-correctness",
      stepName: "compare-postgres",
      focusedCommand: ["pgx-cli", "test", "compare-postgres", "--workload", "tpch-correctness"]
    }
  ])("review $stepName failure records runnable focused command", async ({ match, stepName, focusedCommand, beforeFailure = [] }) => {
    const runner = new FakeRunner();
    runner.streamingResults.push(...beforeFailure, { match, exitCode: 1, stderr: `${stepName} failed\n` });
    const output = { stdout: "", stderr: "" };
    const config = makeDevConfig();

    const exitCode = await runDevCommand(["gate", "review"], runner, output, config);

    expect(exitCode).toBe(1);
    const state = JSON.parse(readFileSync(gateFailureStatePath(config.localProjectPath), "utf8"));
    expect(state).toMatchObject({ stepName, focusedCommand });
    expect(state.focusedCommand).not.toEqual(["pgx-cli", "dev", "lint", "all"]);
    expect(state.focusedCommand).not.toEqual(["pgx-cli", "dev", "test", "all"]);
  });

  test("immediate review rerun without override blocks and prints focused command", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = makeDevConfig();
    recordGateFailure({
      root: config.localProjectPath,
      gate: "review",
      head: "abc123",
      stepName: "lint",
      stepCommand: ["pgx-cli", "dev", "lint", "diff"],
      runId: "run-1"
    });

    const exitCode = await runDevCommand(["gate", "review"], runner, output, config);

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("run focused reproducer first");
    expect(output.stderr).toContain("pgx-cli dev lint diff");
    expect(output.stderr).not.toContain("pgx-cli dev lint all");
    expect(runner.calls.some((call) => call.command === "ssh")).toBe(false);
  });

  test("--rerun-full allows review rerun and prints override note", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = makeDevConfig();
    recordGateFailure({
      root: config.localProjectPath,
      gate: "review",
      head: "abc123",
      stepName: "utest-pg",
      stepCommand: ["pgx-cli", "dev", "test", "focused"],
      runId: "run-1"
    });

    const exitCode = await runDevCommand(["gate", "review", "--rerun-full"], runner, output, config);

    expect(exitCode).toBe(0);
    expect(output.stderr).toContain("full review gate override accepted");
    expect(runner.calls.some((call) => call.command === "ssh")).toBe(true);
  });

  test("successful matching non-test command clears review gate block", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = makeDevConfig();
    recordGateFailure({
      root: config.localProjectPath,
      gate: "review",
      head: "abc123",
      stepName: "lint",
      stepCommand: ["pgx-cli", "dev", "lint", "diff"],
      runId: "run-1"
    });

    const exitCode = await runDevCommand(["lint", "diff"], runner, output, config);

    expect(exitCode).toBe(0);
    expect(existsSync(gateFailureStatePath(config.localProjectPath))).toBe(false);
    expect(output.stdout).toContain("focused reproducer passed");
  });

  test("successful focused command clears matching review gate block", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = makeDevConfig();
    recordGateFailure({
      root: config.localProjectPath,
      gate: "review",
      head: "abc123",
      stepName: "utest-pg",
      stepCommand: ["pgx-cli", "dev", "test", "focused"],
      runId: "run-1"
    });

    const exitCode = await runDevCommand(["test", "focused"], runner, output, config);

    expect(exitCode).toBe(0);
    expect(existsSync(gateFailureStatePath(config.localProjectPath))).toBe(false);
    expect(output.stdout).toContain("focused reproducer passed");
  });

  test("dev gate batch does not run compare-postgres", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["gate", "batch"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).not.toContain("compare-postgres");
  });
});
