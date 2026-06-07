import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { RunResult, StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { formatWorkflowSummary, runDevCommand } from "../src/dev.js";

class FakeRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];
  results: RunResult[] = [];

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    if (command === "mutagen" && args[1] === "list") {
      return { exitCode: 0, stdout: healthyJson(), stderr: "" };
    }
    const rendered = [command, ...args].join(" ");
    if (rendered.includes("pg_regress") || rendered.includes("ctest -V")) {
      return { exitCode: 0, stdout: "1: ok 1 - 1_one_tuple 10 ms\n", stderr: "" };
    }
    return this.results.shift() ?? { exitCode: 0, stdout: "", stderr: "" };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args });
    const rendered = [command, ...args].join(" ");
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
  mkdirSync(join(root, "src/pgx-lower/test"), { recursive: true });
  mkdirSync(join(root, "tests"), { recursive: true });
  writeFileSync(join(root, "src/pgx-lower/test/type_mapping_tests.cpp"), "PGX_TEST_FN(type_mapping_smoke) {}\n");
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

  test.each([
    [["lint", "diff"], "ssh", "clang-tidy-diff-20"],
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
    expect(extensionCmake).toContain("pgx_lower_regress_unit");
    expect(extensionCmake).toContain("route-check");
    expect(extensionCmake).toContain("tests/unit-tests/sql");
    expect(extensionCmake).toContain("test unit-sql");
    expect(extensionCmake).toContain("pgx-cli/dist/index.js");
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
    const exitCode = await runDevCommand(["gate", "review"], runner, output, makeDevConfig());

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
    const waitIndex = commands.indexOf('tsp -w "$id"');
    const catIndex = commands.indexOf('tsp -c "$id"');
    expect(waitIndex).toBeGreaterThan(-1);
    expect(catIndex).toBeGreaterThan(waitIndex);
    expect(commands).not.toContain(oldBaselineScript);
    expect(commands).not.toContain("just");
  });

  test("dev gate review no-bench does not run bench", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["gate", "review", "--no-bench"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    expect(output.stdout).not.toContain("bench");
  });
});
