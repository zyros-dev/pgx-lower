import { describe, expect, test } from "vitest";
import { NodeCommandRunner } from "../src/commands.js";
import {
  runQueueCommand,
  runSetupCommand,
  runSyncCommand,
  runThorCommand
} from "../src/operations.js";
import type { CommandRunner, RunResult } from "../src/commands.js";

class FakeRunner implements CommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];
  results: RunResult[] = [];

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    return this.results.shift() ?? { exitCode: 0, stdout: "", stderr: "" };
  }
}

const thorConfig = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower"
};

const setupConfig = {
  ...thorConfig,
  packageDir: "/Users/nickvandermerwe/repos/pgx-lower/pgx-cli",
  dockerContainer: "pgx-lower-dev"
};

describe("operations", () => {
  test("node command runner can be constructed", () => {
    expect(new NodeCommandRunner()).toBeInstanceOf(NodeCommandRunner);
  });

  test("fake runner records commands", async () => {
    const runner = new FakeRunner();
    await runner.run("mutagen", ["sync", "list", "pgx-lower"]);
    expect(runner.calls).toEqual([
      { command: "mutagen", args: ["sync", "list", "pgx-lower"] }
    ]);
  });

  test("setup install builds and links the in-repo package", async () => {
    const runner = new FakeRunner();
    runner.results = [
      { exitCode: 0, stdout: "/usr/local/bin/pgx-cli\n", stderr: "" },
      { exitCode: 0, stdout: "", stderr: "" },
      { exitCode: 0, stdout: "", stderr: "" },
      { exitCode: 0, stdout: "", stderr: "" }
    ];
    const output = { stdout: "", stderr: "" };
    const exitCode = await runSetupCommand(["install"], runner, output, setupConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "sh", args: ["-lc", "command -v pgx-cli || true"] },
      { command: "npm", args: ["--prefix", "/Users/nickvandermerwe/repos/pgx-lower/pgx-cli", "install"] },
      { command: "npm", args: ["--prefix", "/Users/nickvandermerwe/repos/pgx-lower/pgx-cli", "run", "build"] },
      { command: "sh", args: ["-lc", "cd /Users/nickvandermerwe/repos/pgx-lower/pgx-cli && npm link --force"] }
    ]);
    expect(output.stdout).toContain("Existing pgx-cli: /usr/local/bin/pgx-cli");
  });

  test("setup doctor checks the host workflow dependencies", async () => {
    const runner = new FakeRunner();
    runner.results = [
      { exitCode: 0, stdout: "/usr/local/bin/pgx-cli\n", stderr: "" },
      { exitCode: 0, stdout: "Session: pgx-lower\n", stderr: "" },
      { exitCode: 0, stdout: "", stderr: "" },
      { exitCode: 0, stdout: "", stderr: "" },
      { exitCode: 0, stdout: "/usr/bin/tsp\n", stderr: "" },
      { exitCode: 0, stdout: "pgx-lower-dev\n", stderr: "" }
    ];
    const output = { stdout: "", stderr: "" };
    const exitCode = await runSetupCommand(["doctor"], runner, output, setupConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "sh", args: ["-lc", "command -v pgx-cli || true"] },
      { command: "mutagen", args: ["sync", "list", "pgx-lower"] },
      { command: "ssh", args: ["comfy", "true"] },
      { command: "ssh", args: ["comfy", "test", "-d", "/home/zel/repos/pgx-lower"] },
      { command: "ssh", args: ["comfy", "command", "-v", "tsp"] },
      { command: "ssh", args: ["comfy", "docker", "ps", "--format", "{{.Names}}"] }
    ]);
    expect(output.stdout).toContain("setup doctor: ok");
  });

  test("sync status lists the configured mutagen session", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runSyncCommand(["status"], runner, output, {
      mutagenSession: "pgx-lower"
    });

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "mutagen", args: ["sync", "list", "pgx-lower"] }
    ]);
    expect(output.stdout).toContain("Mutagen session: pgx-lower");
  });

  test("sync flush flushes the configured mutagen session", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runSyncCommand(["flush"], runner, output, {
      mutagenSession: "pgx-lower"
    });

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "mutagen", args: ["sync", "flush", "pgx-lower"] }
    ]);
  });

  test("thor rejects retired just passthrough", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runThorCommand(["just", "compile"], runner, output, thorConfig);

    expect(exitCode).toBe(1);
    expect(runner.calls).toEqual([]);
    expect(output.stderr).toContain("Usage: thor shell --dangerous -- <cmd...>");
  });

  test("thor shell requires dangerous flag", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runThorCommand(["shell", "--", "git", "status"], runner, output, thorConfig);

    expect(exitCode).toBe(1);
    expect(runner.calls).toEqual([]);
    expect(output.stderr).toContain("--dangerous");
  });

  test("thor shell dangerous flushes then runs shell command on thor", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runThorCommand(
      ["shell", "--dangerous", "--", "git", "status", "--short"],
      runner,
      output,
      thorConfig
    );

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "mutagen", args: ["sync", "flush", "pgx-lower"] },
      {
        command: "ssh",
        args: [
          "comfy",
          "bash",
          "-lc",
          "'export PATH=$HOME/.local/bin:$PATH && cd /home/zel/repos/pgx-lower && git status --short'"
        ]
      }
    ]);
  });

  test("queue status checks task-spooler queues directly on thor", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runQueueCommand(["status"], runner, output, thorConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "mutagen", args: ["sync", "flush", "pgx-lower"] },
      {
        command: "ssh",
        args: [
          "comfy",
          "bash",
          "-lc",
          "'export PATH=$HOME/.local/bin:$PATH && cd /home/zel/repos/pgx-lower && for q in pgx-build pgx-check; do echo \"=== ${q} queue ===\"; TS_SOCKET=/tmp/${q}.sock tsp; done'"
        ]
      }
    ]);
  });

  test.each([
    [["tail", "7"], "TS_SOCKET=/tmp/pgx-build.sock tsp -t 7"],
    [["cancel", "7"], "TS_SOCKET=/tmp/pgx-build.sock tsp -k 7 || true; TS_SOCKET=/tmp/pgx-build.sock tsp -r 7"]
  ])("queue %s runs task-spooler maintenance directly on thor", async (args, shellCommand) => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runQueueCommand(args, runner, output, thorConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "mutagen", args: ["sync", "flush", "pgx-lower"] },
      {
        command: "ssh",
        args: [
          "comfy",
          "bash",
          "-lc",
          `'export PATH=$HOME/.local/bin:$PATH && cd /home/zel/repos/pgx-lower && ${shellCommand}'`
        ]
      }
    ]);
  });

  test("queue flush clears completed build and check queue entries on thor", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runQueueCommand(["flush"], runner, output, thorConfig);

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      { command: "mutagen", args: ["sync", "flush", "pgx-lower"] },
      {
        command: "ssh",
        args: [
          "comfy",
          "bash",
          "-lc",
          "'export PATH=$HOME/.local/bin:$PATH && cd /home/zel/repos/pgx-lower && TS_SOCKET=/tmp/pgx-build.sock tsp -C && TS_SOCKET=/tmp/pgx-check.sock tsp -C'"
        ]
      }
    ]);
  });

  test("queue tail requires an id", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runQueueCommand(["tail"], runner, output, thorConfig);

    expect(exitCode).toBe(1);
    expect(runner.calls).toEqual([]);
    expect(output.stderr).toContain("Usage: queue");
  });

  test("queue cancel rejects non-numeric ids", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runQueueCommand(["cancel", "$(rm", "-rf"], runner, output, thorConfig);

    expect(exitCode).toBe(1);
    expect(runner.calls).toEqual([]);
    expect(output.stderr).toContain("Usage: queue");
  });
});
