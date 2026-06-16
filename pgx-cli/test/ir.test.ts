import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { runIrCommand } from "../src/ir.js";

class IrRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];

  constructor(private readonly result: { exitCode?: number; stdout?: string; stderr?: string } = {}) {}

  async run() {
    return { exitCode: 0, stdout: "", stderr: "" };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args });
    const rendered = [command, ...args].join(" ");
    const stdout = command === "mutagen" && args[1] === "list"
      ? JSON.stringify([{ name: "pgx-lower", paused: false, status: "watching", alpha: { connected: true }, beta: { connected: true } }])
      : this.result.stdout ?? "ir line\n";
    const stderr = command === "mutagen" ? "" : this.result.stderr ?? "";
    options.stdout?.write(stdout);
    options.stderr?.write(stderr);
    return {
      childExitCode: command === "mutagen" ? 0 : this.result.exitCode ?? 0,
      stdoutSample: { head: stdout, tail: "", truncated: false },
      stderrSample: { head: stderr, tail: "", truncated: false },
      timedOut: false
    };
  }
}

function irConfig(root: string, fullOutputRequiresFlag = true) {
  return {
    localProjectPath: root,
    mutagenSession: "pgx-lower",
    sshHost: "comfy",
    remoteProjectPath: "/home/zel/repos/pgx-lower",
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
      full_output_requires_flag: fullOutputRequiresFlag
    }
  } as const;
}

describe("ir command", () => {
  test("inspect latest tail prints a bounded preview and transcript path", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-ir-"));
    try {
      const runner = new IrRunner();
      const output = { stdout: "", stderr: "" };

      const exitCode = await runIrCommand(["inspect", "latest", "--tail", "20"], runner, output, irConfig(root));

      expect(exitCode).toBe(0);
      const command = runner.calls.at(-1)?.args.join(" ") ?? "";
      expect(command).toContain("find /tmp/pgx_ir");
      expect(command).toContain("tail -n 20");
      expect(command).not.toMatch(/\bcat\b/);
      expect(output.stdout).toContain("ir line");
      expect(output.stdout).toContain("transcript:");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("inspect pattern prints matching snippets only", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-ir-pattern-"));
    try {
      const runner = new IrRunner();
      const output = { stdout: "", stderr: "" };

      const exitCode = await runIrCommand(["inspect", "/tmp/pgx_ir/file.ir", "--pattern", "db.cast"], runner, output, irConfig(root));

      expect(exitCode).toBe(0);
      const command = runner.calls.at(-1)?.args.join(" ") ?? "";
      expect(command).toContain("rg -n --color never");
      expect(command).toContain("db.cast");
      expect(command).not.toMatch(/\bcat\b/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("missing /tmp/pgx_ir produces an actionable error", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-ir-missing-"));
    try {
      const runner = new IrRunner({
        exitCode: 1,
        stdout: "",
        stderr: "missing /tmp/pgx_ir\nnext: run a pgx-lower workflow that emits IR, then retry pgx-cli ir inspect latest\n"
      });
      const output = { stdout: "", stderr: "" };

      const exitCode = await runIrCommand(["inspect", "latest", "--tail", "20"], runner, output, irConfig(root));

      expect(exitCode).toBe(1);
      expect(output.stdout).toContain("missing /tmp/pgx_ir");
      expect(output.stdout).toContain("next: run a pgx-lower workflow that emits IR");
      const runId = output.stdout.match(/run id: ([^\n]+)/)?.[1];
      expect(runId).toEqual(expect.any(String));
      const command = runner.calls.at(-1)?.args.join(" ") ?? "";
      expect(command).toContain("missing /tmp/pgx_ir");
      expect(command).toContain("next: run a pgx-lower workflow that emits IR");
      const summary = JSON.parse(readFileSync(join(root, ".pgx-cli", "runs", runId ?? "", "summary.json"), "utf8"));
      expect(summary.commandName).toBe("ir-inspect");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("allows explicit full output when full output requires a flag", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-ir-full-"));
    try {
      const runner = new IrRunner();
      const output = { stdout: "", stderr: "" };

      const exitCode = await runIrCommand(["inspect", "latest", "--full"], runner, output, irConfig(root));

      expect(exitCode).toBe(0);
      const command = runner.calls.at(-1)?.args.join(" ") ?? "";
      expect(command).toContain("sed -n");
      expect(command).toContain("1,$p");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("allows full output when the config disables full-output protection", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-ir-full-allowed-"));
    try {
      const runner = new IrRunner();
      const output = { stdout: "", stderr: "" };

      const exitCode = await runIrCommand(["inspect", "latest", "--full"], runner, output, irConfig(root, false));

      expect(exitCode).toBe(0);
      const command = runner.calls.at(-1)?.args.join(" ") ?? "";
      expect(command).toContain("sed -n");
      expect(command).toContain("1,$p");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
