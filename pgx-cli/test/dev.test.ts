import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { CommandRunner, RunResult } from "../src/commands.js";
import { formatWorkflowSummary, runDevCommand } from "../src/dev.js";

class FakeRunner implements CommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];
  results: RunResult[] = [];

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    const rendered = [command, ...args].join(" ");
    if (rendered.includes("pg_regress") || rendered.includes("ctest -V")) {
      return { exitCode: 0, stdout: "1: ok 1 - 1_one_tuple 10 ms\n", stderr: "" };
    }
    return this.results.shift() ?? { exitCode: 0, stdout: "", stderr: "" };
  }
}

const devConfigBase = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower",
  dockerContainer: "pgx-lower-dev",
  buildQueue: "pgx-build",
  checkQueue: "pgx-check"
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
    expect(runner.calls).toEqual([
      { command: "mutagen", args: ["sync", "list", "pgx-lower"] },
      { command: "ssh", args: ["comfy", "bash", "-lc", "'cd /home/zel/repos/pgx-lower && git status --short --branch'"] },
      { command: "ssh", args: ["comfy", "bash", "-lc", "'TS_SOCKET=/tmp/pgx-build.sock tsp'"] },
      { command: "ssh", args: ["comfy", "bash", "-lc", "'TS_SOCKET=/tmp/pgx-check.sock tsp'"] }
    ]);
    expect(output.stdout).toContain("dev status");
  });

  test("dev logs latest tails the build queue", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["logs", "latest"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "ssh", args: ["comfy", "bash", "-lc", "'TS_SOCKET=/tmp/pgx-build.sock tsp -t'"] }
    ]);
  });

  test("dev logs id tails a specific build queue job", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["logs", "7"], runner, output, makeDevConfig());

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "ssh", args: ["comfy", "bash", "-lc", "'TS_SOCKET=/tmp/pgx-build.sock tsp -t 7'"] }
    ]);
  });

  test.each([
    [["lint", "diff"], "sh", "clang-tidy-diff-20"],
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
    expect(runner.calls.some((call) => call.command === "python3")).toBe(false);
    expect(commands).not.toContain(oldLintScript);
    expect(commands).not.toContain(oldBaselineScript);
    expect(commands).not.toContain("just");
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
    expect(commands).toContain("pgx-compile.out");
    expect(commands).toContain("UTEST-PG_OK");
    expect(commands).toContain("ctest -V");
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
