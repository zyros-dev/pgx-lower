import { describe, expect, test } from "vitest";
import { tmpdir } from "node:os";
import type { RunResult, StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { runDockerCommand } from "../src/docker.js";

const oldPtestWrapper = ["build", "ptest.sh"].join("-");
const oldReleaseWrapper = ["build", "release.sh"].join("-");

class FakeRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    return { exitCode: 0, stdout: "", stderr: "" };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args });
    const stdout = command === "mutagen" && args[1] === "list"
      ? JSON.stringify([{ name: "pgx-lower", paused: false, status: "watching", alpha: { connected: true }, beta: { connected: true } }])
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

const config = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower",
  localProjectPath: tmpdir(),
  dockerContainer: "pgx-lower-dev",
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
    max_lines_total: 60,
    failure_tail_lines: 20,
    success_tail_lines: 10,
    progress: "final-summary",
    full_output_requires_flag: true
  }
};

describe("docker commands", () => {
  test("docker status checks the dev container on thor", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDockerCommand(["status"], runner, output, config);

    expect(exitCode).toBe(0);
    expect(runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n")).toContain("docker ps");
    expect(output.stdout).toContain("run id:");
  });

  test("docker build ptest replaces the old ptest wrapper", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDockerCommand(["build", "ptest"], runner, output, config);

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("npm --prefix pgx-cli install && npm --prefix pgx-cli run build");
    expect(commands).toContain("build-artifacts/docker/ptest");
    expect(commands).toContain("ctest --output-on-failure");
    expect(commands).not.toContain(oldPtestWrapper);
  });

  test("docker build release replaces the old release wrapper", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDockerCommand(["build", "release"], runner, output, config);

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("npm --prefix pgx-cli install && npm --prefix pgx-cli run build");
    expect(commands).toContain("RelWithDebInfo");
    expect(commands).toContain("strip --strip-debug");
    expect(commands).not.toContain(oldReleaseWrapper);
  });
});
