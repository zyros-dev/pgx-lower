import { describe, expect, test } from "vitest";
import type { CommandRunner, RunResult } from "../src/commands.js";
import { runLogsCommand } from "../src/logs.js";

class FakeRunner implements CommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    return { exitCode: 0, stdout: "line\n", stderr: "" };
  }
}

const config = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower",
  dockerContainer: "pgx-lower-dev"
};

describe("logs commands", () => {
  test("logs errors tails the pgx error log inside the dev container", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runLogsCommand(["errors", "--lines", "50"], runner, output, config);

    expect(exitCode).toBe(0);
    expect(output.stdout).toContain("line");
    expect(runner.calls).toEqual([
      {
        command: "ssh",
        args: [
          "comfy",
          "bash",
          "-lc",
          "'cd /home/zel/repos/pgx-lower && docker exec pgx-lower-dev tail -n 50 /tmp/pgx_errors.log'"
        ]
      }
    ]);
  });

  test("logs file quotes paths before tailing them inside the dev container", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runLogsCommand(["file", "/tmp/path with spaces.log"], runner, output, {
      ...config,
      runningOnRemote: true
    });

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      {
        command: "bash",
        args: [
          "-lc",
          "cd /home/zel/repos/pgx-lower && docker exec pgx-lower-dev tail -n 50 '/tmp/path with spaces.log'"
        ]
      }
    ]);
  });

  test("logs docker tails the dev container log", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runLogsCommand(["docker", "--lines", "80"], runner, output, config);

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([
      {
        command: "ssh",
        args: ["comfy", "bash", "-lc", "'cd /home/zel/repos/pgx-lower && docker logs --tail 80 pgx-lower-dev'"]
      }
    ]);
  });

  test("logs rejects non-positive line counts", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runLogsCommand(["errors", "--lines", "0"], runner, output, config);

    expect(exitCode).toBe(1);
    expect(runner.calls).toEqual([]);
    expect(output.stderr).toContain("Usage: logs");
  });
});
