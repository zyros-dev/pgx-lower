import { describe, expect, test } from "vitest";
import type { CommandRunner, RunResult } from "../src/commands.js";
import { runDockerCommand } from "../src/docker.js";

const oldPtestWrapper = ["build", "ptest.sh"].join("-");
const oldReleaseWrapper = ["build", "release.sh"].join("-");

class FakeRunner implements CommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    return { exitCode: 0, stdout: "", stderr: "" };
  }
}

const config = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower",
  dockerContainer: "pgx-lower-dev"
};

describe("docker commands", () => {
  test("docker status checks the dev container on thor", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDockerCommand(["status"], runner, output, config);

    expect(exitCode).toBe(0);
    expect(runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n")).toContain("docker ps");
  });

  test("docker build ptest replaces the old ptest wrapper", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDockerCommand(["build", "ptest"], runner, output, config);

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
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
    expect(commands).toContain("RelWithDebInfo");
    expect(commands).toContain("strip --strip-debug");
    expect(commands).not.toContain(oldReleaseWrapper);
  });
});
