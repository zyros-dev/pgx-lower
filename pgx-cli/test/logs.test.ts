import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { runLogsCommand } from "../src/logs.js";
import type { StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";

const noopRunner: StreamingCommandRunner = {
  async run() {
    return { exitCode: 0, stdout: "", stderr: "" };
  },
  async runStreaming() {
    return {
      childExitCode: 0,
      stdoutSample: { head: "", tail: "", truncated: false },
      stderrSample: { head: "", tail: "", truncated: false },
      timedOut: false
    };
  }
};

class RemoteLogsRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];

  async run() {
    return { exitCode: 0, stdout: "", stderr: "" };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args });
    const stdout = command === "mutagen" && args[1] === "list"
      ? JSON.stringify([{ name: "pgx-lower", paused: false, status: "watching", alpha: { connected: true }, beta: { connected: true } }])
      : command === "ssh"
        ? "log line\n"
        : "";
    options.stdout?.write(stdout);
    return {
      childExitCode: 0,
      stdoutSample: { head: stdout, tail: "", truncated: false },
      stderrSample: { head: "", tail: "", truncated: false },
      timedOut: false
    };
  }
}

function remoteLogsConfig(root: string) {
  return {
    localProjectPath: root,
    mutagenSession: "pgx-lower",
    sshHost: "comfy",
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
      max_lines_per_step: 80,
      max_lines_total: 180,
      failure_tail_lines: 60,
      success_tail_lines: 20,
      progress: "final-summary",
      full_output_requires_flag: true
    }
  } as const;
}

describe("logs command", () => {
  test("shows a bounded combined transcript by run id", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-logs-"));
    try {
      const runDir = join(root, ".pgx-cli", "runs", "2026-run");
      mkdirSync(runDir, { recursive: true });
      writeFileSync(join(runDir, "combined.log"), Array.from({ length: 30 }, (_, index) => `line-${index}`).join("\n"));
      writeFileSync(join(runDir, "summary.json"), JSON.stringify({ runId: "2026-run", workflowExitCode: 1 }));
      const output = { stdout: "", stderr: "" };

      const exitCode = await runLogsCommand(["show", "2026-run", "--tail", "3"], noopRunner, output, {
        localProjectPath: root,
        output: {
          mode: "agent",
          transcript_dir: ".pgx-cli/runs",
          max_lines_per_step: 80,
          max_lines_total: 180,
          failure_tail_lines: 60,
          success_tail_lines: 20,
          progress: "final-summary",
          full_output_requires_flag: true
        }
      });

      expect(exitCode).toBe(0);
      expect(output.stdout).toContain("run id: 2026-run");
      expect(output.stdout).toContain("line-29");
      expect(output.stdout).not.toContain("line-1\nline-2\nline-3\nline-4");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("latest resolves the newest run directory", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-logs-latest-"));
    try {
      for (const id of ["2026-older", "2026-newer"]) {
        const runDir = join(root, ".pgx-cli", "runs", id);
        mkdirSync(runDir, { recursive: true });
        writeFileSync(join(runDir, "combined.log"), `${id}\n`);
        writeFileSync(join(runDir, "summary.json"), JSON.stringify({ runId: id }));
      }
      const output = { stdout: "", stderr: "" };

      const exitCode = await runLogsCommand(["latest", "--tail", "5"], noopRunner, output, {
        localProjectPath: root,
        output: {
          mode: "agent",
          transcript_dir: ".pgx-cli/runs",
          max_lines_per_step: 80,
          max_lines_total: 180,
          failure_tail_lines: 60,
          success_tail_lines: 20,
          progress: "final-summary",
          full_output_requires_flag: true
        }
      });

      expect(exitCode).toBe(0);
      expect(output.stdout).toContain("run id: 2026-newer");
      expect(output.stdout).toContain("2026-newer");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("shows a bounded combined transcript head by run id", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-logs-head-"));
    try {
      const runDir = join(root, ".pgx-cli", "runs", "2026-run");
      mkdirSync(runDir, { recursive: true });
      writeFileSync(join(runDir, "combined.log"), Array.from({ length: 10 }, (_, index) => `line-${index}`).join("\n"));
      const output = { stdout: "", stderr: "" };

      const exitCode = await runLogsCommand(["show", "2026-run", "--head", "2"], noopRunner, output, {
        localProjectPath: root,
        output: {
          mode: "agent",
          transcript_dir: ".pgx-cli/runs",
          max_lines_per_step: 80,
          max_lines_total: 180,
          failure_tail_lines: 60,
          success_tail_lines: 20,
          progress: "final-summary",
          full_output_requires_flag: true
        }
      });

      expect(exitCode).toBe(0);
      expect(output.stdout).toContain("line-0\nline-1\n");
      expect(output.stdout).not.toContain("line-9");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test.each([
    ["errors", ["errors", "--tail", "1"]],
    ["docker", ["docker", "--tail", "1"]],
    ["file", ["file", "/tmp/pgx.log"]]
  ])("remote logs %s records explicit sync proof skip reason", async (_name, args) => {
    const root = mkdtempSync(join(tmpdir(), "pgx-logs-remote-"));
    try {
      const runner = new RemoteLogsRunner();
      const output = { stdout: "", stderr: "" };

      const exitCode = await runLogsCommand(args, runner, output, remoteLogsConfig(root));

      expect(exitCode).toBe(0);
      expect(output.stdout).toContain("sync proof: skipped (remote-only logs command)");
      const runId = output.stdout.match(/run id: ([^\n]+)/)?.[1];
      expect(runId).toEqual(expect.any(String));
      const summary = JSON.parse(readFileSync(join(root, ".pgx-cli", "runs", runId ?? "", "summary.json"), "utf8"));
      expect(summary.mutagenPreflight.proofSkippedReason).toBe("remote-only logs command");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
