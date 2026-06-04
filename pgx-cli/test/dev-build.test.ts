import { expect, test } from "vitest";
import type { CommandRunner } from "../src/commands.js";
import { runDevBuildCommand } from "../src/dev-build.js";

class FakeRunner implements CommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];

  async run(command: string, args: string[]) {
    this.calls.push({ command, args });
    return { stdout: "", stderr: "", exitCode: 0 };
  }
}

const devBuildConfig = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower",
  dockerContainer: "pgx-lower-dev",
  profileName: "debug",
  profile: {
    build: {
      build_dir: "build-artifacts/ptest",
      cmake: { generator: "Ninja", args: { CMAKE_BUILD_TYPE: "Debug" } }
    },
    runtime: { logging: "info" }
  }
};

test("dev build explain prints resolved profile without remote execution", async () => {
  const runner = new FakeRunner();
  const output = { stdout: "", stderr: "" };
  const exitCode = await runDevBuildCommand(["explain", "--profile", "debug"], runner, output, devBuildConfig);

  expect(exitCode).toBe(0);
  expect(runner.calls).toEqual([]);
  expect(output.stdout).toContain("profile: debug");
});

test("dev build configure runs cmake in the configured docker container", async () => {
  const runner = new FakeRunner();
  const output = { stdout: "", stderr: "" };
  const exitCode = await runDevBuildCommand(["configure", "--profile", "debug"], runner, output, devBuildConfig);

  expect(exitCode).toBe(0);
  expect(runner.calls.at(-1)).toEqual({
    command: "ssh",
    args: [
      "comfy",
      "bash",
      "-lc",
      "'cd /home/zel/repos/pgx-lower && docker exec pgx-lower-dev cmake -S /workspace -B build-artifacts/ptest -G Ninja -DCMAKE_BUILD_TYPE=Debug'"
    ]
  });
});
