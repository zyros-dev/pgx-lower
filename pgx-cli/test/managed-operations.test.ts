import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { runLogsCommand } from "../src/logs.js";
import { runManagedRemoteShell } from "../src/managed-operations.js";
import type { ManagedOperationConfig } from "../src/managed-operations.js";

class FakeRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[]; options: StreamingRunOptions }> = [];
  results: Array<{ exitCode: number; stdout: string; stderr?: string }> = [];

  async run(command: string, args: string[]) {
    const result = await this.runStreaming(command, args, {});
    return { exitCode: result.childExitCode, stdout: result.stdoutSample.head, stderr: result.stderrSample.head };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args, options });
    const result = this.results.shift() ?? { exitCode: 0, stdout: healthyJson() };
    options.stdout?.write(result.stdout);
    options.stderr?.write(result.stderr ?? "");
    return {
      childExitCode: result.exitCode,
      stdoutSample: { head: result.stdout, tail: "", truncated: false },
      stderrSample: { head: result.stderr ?? "", tail: "", truncated: false },
      timedOut: false
    };
  }
}

function healthyJson(): string {
  return JSON.stringify([{ name: "pgx-lower", paused: false, status: "watching", alpha: { connected: true }, beta: { connected: true } }]);
}

function makeConfig(root: string): ManagedOperationConfig {
  return {
    mutagenSession: "pgx-lower",
    sshHost: "comfy",
    localProjectPath: root,
    remoteProjectPath: "/home/zel/repos/pgx-lower",
    dockerContainer: "pgx-lower-dev",
    sync: {
      required_for_remote: true,
      flush_timeout_seconds: 45,
      proof: {
        enabled: true,
        path: ".pgx-cli/sync-probes",
        required_for: ["build", "test", "lint", "psql", "pg_regress", "bench", "profile", "run"]
      }
    },
    output: {
      mode: "agent",
      transcript_dir: ".pgx-cli/runs",
      max_lines_per_step: 4,
      max_lines_total: 12,
      failure_tail_lines: 3,
      success_tail_lines: 2,
      progress: "final-summary",
      full_output_requires_flag: true
    }
  };
}

describe("managed operations", () => {
  test("runs mutagen preflight before ssh and emits bounded output", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-managed-op-"));
    try {
      const runner = new FakeRunner();
      runner.results = [
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "line\n".repeat(50) }
      ];
      const output = { stdout: "", stderr: "" };

      const result = await runManagedRemoteShell({
        runner,
        output,
        config: makeConfig(root),
        commandName: "dev-test-unit-pg_nullability",
        shellCommand: "true",
        requireMutagenProof: false,
        metadata: {
          profile: { name: "debug" }
        },
        artifactPaths: ["/workspace/build-artifacts/ptest"]
      });

      expect(result.workflowExitCode).toBe(0);
      expect(runner.calls.map((call) => call.command)).toEqual(["mutagen", "mutagen", "mutagen", "ssh"]);
      expect(output.stdout).toContain("run id:");
      expect(output.stdout).toContain("transcript:");
      expect(output.stdout).toContain("omitted");
      expect(output.stdout.split("\n").length).toBeLessThanOrEqual(14);
      const summary = JSON.parse(readFileSync(result.artifact.summaryPath, "utf8"));
      expect(summary.target).toEqual({
        kind: "ssh",
        sshHost: "comfy",
        remoteProjectPath: "/home/zel/repos/pgx-lower",
        dockerContainer: "pgx-lower-dev"
      });
      expect(summary.profile).toEqual({ name: "debug" });
      expect(summary.artifactPaths).toEqual(["/workspace/build-artifacts/ptest"]);
      expect(readFileSync(result.artifact.artifactsPath, "utf8")).toContain("/workspace/build-artifacts/ptest");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("uses non-login bash for remote shell commands", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-managed-non-login-"));
    try {
      const runner = new FakeRunner();
      runner.results = [
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "ok\n" }
      ];
      const output = { stdout: "", stderr: "" };

      await runManagedRemoteShell({
        runner,
        output,
        config: makeConfig(root),
        commandName: "dev-check-diff",
        shellCommand: "set -e; echo clean; exit 0",
        requireMutagenProof: false
      });

      const sshCall = runner.calls.find((call) => call.command === "ssh");
      expect(sshCall?.args.slice(0, 3)).toEqual(["comfy", "bash", "-c"]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("blocks before ssh when mutagen health fails", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-managed-block-"));
    try {
      const runner = new FakeRunner();
      runner.results = [{ exitCode: 0, stdout: "[]" }];
      const output = { stdout: "", stderr: "" };

      const result = await runManagedRemoteShell({
        runner,
        output,
        config: makeConfig(root),
        commandName: "dev-test-unit-pg_nullability",
        shellCommand: "true",
        requireMutagenProof: true
      });

      expect(result.workflowExitCode).toBe(1);
      expect(runner.calls.map((call) => call.command)).toEqual(["mutagen"]);
      expect(output.stderr).toContain("blocked before running");
      expect(output.stderr).toContain("pgx-cli sync doctor");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("preflight-blocked runs still write standard artifacts and readable logs", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-managed-block-artifacts-"));
    try {
      const runner = new FakeRunner();
      runner.results = [{ exitCode: 0, stdout: "[]" }];
      const output = { stdout: "", stderr: "" };

      const result = await runManagedRemoteShell({
        runner,
        output,
        config: makeConfig(root),
        commandName: "run-thor-true",
        shellCommand: "true",
        requireMutagenProof: true,
        artifactPaths: ["/workspace/build-artifacts/ptest"]
      });

      expect(result.workflowExitCode).toBe(1);
      expect(existsSync(result.artifact.commandPath)).toBe(true);
      expect(existsSync(result.artifact.stdoutPath)).toBe(true);
      expect(existsSync(result.artifact.stderrPath)).toBe(true);
      expect(existsSync(result.artifact.combinedPath)).toBe(true);
      expect(readFileSync(result.artifact.artifactsPath, "utf8")).toContain("/workspace/build-artifacts/ptest");
      expect(readFileSync(result.artifact.combinedPath, "utf8")).toContain("blocked before running run-thor-true");

      const summary = JSON.parse(readFileSync(result.artifact.summaryPath, "utf8"));
      expect(summary.command).toEqual([
        "ssh",
        "comfy",
        "bash",
        "-c",
        "'export PATH=$HOME/.local/bin:$PATH && cd /home/zel/repos/pgx-lower && true'"
      ]);
      expect(summary.cwd).toBe(process.cwd());
      expect(summary.startedAt).toEqual(expect.any(String));
      expect(summary.finishedAt).toEqual(expect.any(String));
      expect(summary.durationMs).toEqual(expect.any(Number));
      expect(summary.transcripts).toEqual({
        stdoutPath: result.artifact.stdoutPath,
        stderrPath: result.artifact.stderrPath,
        combinedPath: result.artifact.combinedPath
      });
      expect(summary.artifacts).toEqual({
        registryPath: result.artifact.artifactsPath,
        commandPath: result.artifact.commandPath,
        runDir: result.artifact.runDir
      });
      expect(summary.target).toEqual({
        kind: "ssh",
        sshHost: "comfy",
        remoteProjectPath: "/home/zel/repos/pgx-lower",
        dockerContainer: "pgx-lower-dev"
      });
      expect(summary.project).toEqual({
        localProjectPath: root,
        remoteProjectPath: "/home/zel/repos/pgx-lower",
        mutagenSession: "pgx-lower"
      });
      expect(summary.failureSummary.kind).toBe("mutagen");

      const logsOutput = { stdout: "", stderr: "" };
      const logsExit = await runLogsCommand(["show", result.artifact.runId, "--tail", "20"], runner, logsOutput, makeConfig(root));
      expect(logsExit).toBe(0);
      expect(logsOutput.stdout).toContain(`run id: ${result.artifact.runId}`);
      expect(logsOutput.stdout).toContain("blocked before running run-thor-true");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("managed runs use configured transcript directory for artifacts and logs", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-managed-transcripts-"));
    try {
      const runner = new FakeRunner();
      runner.results = [
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "" },
        { exitCode: 0, stdout: healthyJson() },
        { exitCode: 0, stdout: "ok\n" }
      ];
      const output = { stdout: "", stderr: "" };
      const config = {
        ...makeConfig(root),
        output: {
          ...makeConfig(root).output,
          transcript_dir: "custom-runs"
        }
      };

      const result = await runManagedRemoteShell({
        runner,
        output,
        config,
        commandName: "run-thor-true",
        shellCommand: "true",
        requireMutagenProof: false
      });

      expect(result.workflowExitCode).toBe(0);
      expect(result.artifact.runDir).toBe(join(root, "custom-runs", result.artifact.runId));
      const logsOutput = { stdout: "", stderr: "" };
      const logsExit = await runLogsCommand(["show", result.artifact.runId, "--tail", "5"], runner, logsOutput, config);
      expect(logsExit).toBe(0);
      expect(logsOutput.stdout).toContain("ok");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("running on remote skips ssh and mutagen", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-managed-remote-"));
    try {
      const runner = new FakeRunner();
      runner.results = [{ exitCode: 0, stdout: "ok\n" }];
      const output = { stdout: "", stderr: "" };

      await runManagedRemoteShell({
        runner,
        output,
        config: { ...makeConfig(root), runningOnRemote: true },
        commandName: "queue-status",
        shellCommand: "true",
        requireMutagenProof: true
      });

      expect(runner.calls.map((call) => call.command)).toEqual(["bash"]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
