import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { runBenchCommand } from "../src/bench.js";

class FakeRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[]; options: StreamingRunOptions }> = [];

  async run(command: string, args: string[]) {
    const result = await this.runStreaming(command, args, {});
    return { exitCode: result.childExitCode, stdout: result.stdoutSample.head, stderr: result.stderrSample.head };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args, options });
    const rendered = [command, ...args].join(" ");
    const stdout = command === "mutagen" && args[1] === "list"
      ? JSON.stringify([{ name: "pgx-lower", paused: false, status: "watching", alpha: { connected: true }, beta: { connected: true } }])
      : rendered.includes(".pgx-cli/sync-probes/")
        ? `probe-${rendered.match(new RegExp("sync-probes/([^/']+)\\.txt"))?.[1] ?? "missing"}\n`
        : "bench ok\n";
    options.stdout?.write(stdout);
    return {
      childExitCode: 0,
      stdoutSample: { head: stdout, tail: "", truncated: false },
      stderrSample: { head: "", tail: "", truncated: false },
      timedOut: false
    };
  }
}

function makeConfig(root: string) {
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
      max_lines_per_step: 80,
      max_lines_total: 180,
      failure_tail_lines: 60,
      success_tail_lines: 20,
      progress: "final-summary",
      full_output_requires_flag: true
    }
  } as const;
}

describe("benchmark gateway", () => {
  test("bench tpch runs the migrated helper through managed thor execution", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-bench-"));
    try {
      const runner = new FakeRunner();
      const output = { stdout: "", stderr: "" };

      const exitCode = await runBenchCommand(["tpch", "--", "0.01", "--query", "q01"], runner, output, makeConfig(root));

      expect(exitCode).toBe(0);
      const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
      expect(commands).toContain("python3 benchmark/tpch/run.py 0.01 --query q01");
      expect(commands).toContain("mutagen sync flush pgx-lower");
      expect(commands).toContain(".pgx-cli/sync-probes/");
      expect(output.stdout).toContain("run id:");
      expect(output.stdout).toContain("transcript:");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
