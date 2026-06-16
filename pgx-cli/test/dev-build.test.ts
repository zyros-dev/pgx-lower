import { existsSync, mkdirSync, mkdtempSync } from "node:fs";
import { expect, test } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { runDevBuildCommand } from "../src/dev-build.js";
import { gateFailureStatePath, recordGateFailure } from "../src/gate-memory.js";

class FakeRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];

  async run(command: string, args: string[]) {
    this.calls.push({ command, args });
    if (command === "git" && args.includes("rev-parse")) {
      return { stdout: "abc123\n", stderr: "", exitCode: 0 };
    }
    return { stdout: "", stderr: "", exitCode: 0 };
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

const devBuildConfig = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower",
  localProjectPath: tmpdir(),
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
  },
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
  const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
  expect(commands).toContain("mutagen sync flush pgx-lower");
  expect(commands).toContain("ssh comfy bash -c");
  expect(commands).toContain("npm --prefix pgx-cli install && npm --prefix pgx-cli run build");
  expect(commands.indexOf("npm --prefix pgx-cli install")).toBeLessThan(commands.indexOf("docker exec pgx-lower-dev bash -lc"));
  expect(commands).toContain("rm -f /workspace/CMakeCache.txt");
  expect(commands).toContain("CTestTestfile.cmake");
  expect(commands).toContain("/workspace/src/lingodb/mlir");
  expect(commands).toContain("cmake -S /workspace -B build-artifacts/ptest -G Ninja -DCMAKE_BUILD_TYPE=Debug");
  expect(output.stdout).toContain("run id:");
  expect(output.stdout).toContain("transcript:");
});

test("dev build compile clears a matching review gate block", async () => {
  const root = mkdtempSync(join(tmpdir(), "pgx-dev-build-gate-"));
  mkdirSync(join(root, ".pgx-cli"), { recursive: true });
  recordGateFailure({
    root,
    gate: "review",
    head: "abc123",
    stepName: "compile",
    stepCommand: ["pgx-cli", "dev", "build", "compile", "--profile", "debug"],
    runId: "run-1"
  });
  const runner = new FakeRunner();
  const output = { stdout: "", stderr: "" };

  const exitCode = await runDevBuildCommand(
    ["compile", "--profile", "debug"],
    runner,
    output,
    { ...devBuildConfig, localProjectPath: root }
  );

  expect(exitCode).toBe(0);
  expect(existsSync(gateFailureStatePath(root))).toBe(false);
  expect(output.stdout).toContain("focused reproducer passed");
});
