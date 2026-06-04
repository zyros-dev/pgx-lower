import { describe, expect, test } from "vitest";
import type { CommandRunner, RunResult } from "../src/commands.js";
import { formatWorkflowSummary, runDevCommand } from "../src/dev.js";

class FakeRunner implements CommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];
  results: RunResult[] = [];

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    return this.results.shift() ?? { exitCode: 0, stdout: "", stderr: "" };
  }
}

const devConfig = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower",
  buildQueue: "pgx-build",
  checkQueue: "pgx-check"
};

describe("dev workflow summaries", () => {
  test("formats passing workflow steps", () => {
    expect(
      formatWorkflowSummary([
        { name: "lint diff", command: ["just", "lint-diff"], exitCode: 0, summary: "clean" },
        { name: "compile", command: ["just", "compile"], exitCode: 0, logPath: "/tmp/pgx-compile.out" }
      ])
    ).toBe(
      [
        "Workflow summary:",
        "- ok lint diff: just lint-diff - clean",
        "- ok compile: just compile - log /tmp/pgx-compile.out",
        "Workflow result: ok"
      ].join("\n") + "\n"
    );
  });

  test("formats failing workflow steps", () => {
    expect(
      formatWorkflowSummary([
        { name: "lint diff", command: ["just", "lint-diff"], exitCode: 1, jobId: "12" }
      ])
    ).toContain("- fail lint diff: just lint-diff - job 12");
  });
});

describe("dev commands", () => {
  test("dev status checks sync, branch, and queues", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["status"], runner, output, devConfig);

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
    const exitCode = await runDevCommand(["logs", "latest"], runner, output, devConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "ssh", args: ["comfy", "bash", "-lc", "'TS_SOCKET=/tmp/pgx-build.sock tsp -t'"] }
    ]);
  });

  test("dev logs id tails a specific build queue job", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["logs", "7"], runner, output, devConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "ssh", args: ["comfy", "bash", "-lc", "'TS_SOCKET=/tmp/pgx-build.sock tsp -t 7'"] }
    ]);
  });

  test.each([
    [["lint", "diff"], "just lint-diff"],
    [["lint", "file", "src/pgx-lower/runtime/tuple_access.cpp"], "just lint-files src/pgx-lower/runtime/tuple_access.cpp"],
    [["lint", "files", "a.cpp", "b.cpp"], "just lint-files a.cpp b.cpp"],
    [["test", "unit", "type_mapping"], "just utest-pg-one type_mapping"],
    [["test", "tpch"], "just test-tpch"],
    [["test", "focused"], "just utest-pg"]
  ])("dev %s maps to thor just command", async (args, shellCommand) => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(args, runner, output, devConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "mutagen", args: ["sync", "flush", "pgx-lower"] },
      {
        command: "ssh",
        args: ["comfy", "bash", "-lc", `'cd /home/zel/repos/pgx-lower && ${shellCommand}'`]
      }
    ]);
  });

  test("dev gate batch runs diff-scoped checks", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["gate", "batch"], runner, output, devConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls.filter((call) => call.command === "ssh").map((call) => call.args.at(-1))).toEqual([
      "'cd /home/zel/repos/pgx-lower && just check-diff'",
      "'cd /home/zel/repos/pgx-lower && just lint-diff'",
      "'cd /home/zel/repos/pgx-lower && just utest-pg'"
    ]);
    expect(output.stdout).toContain("Workflow result: ok");
  });

  test("dev gate review runs full review checks", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["gate", "review"], runner, output, devConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls.filter((call) => call.command === "ssh").map((call) => call.args.at(-1))).toEqual([
      "'cd /home/zel/repos/pgx-lower && just check-diff'",
      "'cd /home/zel/repos/pgx-lower && just lint'",
      "'cd /home/zel/repos/pgx-lower && just compile'",
      "'cd /home/zel/repos/pgx-lower && just utest-pg'",
      "'cd /home/zel/repos/pgx-lower && just test'"
    ]);
  });

  test("dev gate review no-bench does not run bench", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevCommand(["gate", "review", "--no-bench"], runner, output, devConfig);

    expect(exitCode).toBe(0);
    expect(output.stdout).not.toContain("bench");
  });
});
