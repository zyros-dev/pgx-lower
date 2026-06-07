import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { runGatewayCommand } from "../src/run.js";
import type { RunGatewayConfig } from "../src/run.js";

class FakeRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[]; options: StreamingRunOptions }> = [];
  commandStdout = "ok\n";

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
        : this.commandStdout;
    options.stdout?.write(stdout);
    return {
      childExitCode: 0,
      stdoutSample: { head: stdout, tail: "", truncated: false },
      stderrSample: { head: "", tail: "", truncated: false },
      timedOut: false
    };
  }
}

function makeConfig(root: string): RunGatewayConfig {
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
  };
}

describe("run gateway", () => {
  test("run thor requires -- and executes through managed ssh", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-run-gateway-"));
    try {
      const runner = new FakeRunner();
      const output = { stdout: "", stderr: "" };
      const exitCode = await runGatewayCommand(["thor", "--", "true"], runner, output, makeConfig(root));

      expect(exitCode).toBe(0);
      const rendered = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
      const sshCalls = runner.calls.filter((call) => call.command === "ssh");
      expect(sshCalls.at(-1)?.args.slice(0, 3)).toEqual(["comfy", "bash", "-c"]);
      expect(rendered).toContain("true");
      expect(output.stdout).toContain("run id:");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("run thor accepts bounded head and tail preview flags", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-run-preview-"));
    try {
      const runner = new FakeRunner();
      runner.commandStdout = Array.from({ length: 8 }, (_, index) => `line-${index}`).join("\n") + "\n";
      const output = { stdout: "", stderr: "" };

      expect(await runGatewayCommand(["thor", "--tail", "5", "--", "true"], runner, output, makeConfig(root))).toBe(0);
      expect(output.stdout).toContain("line-7");
      expect(output.stdout).not.toContain("line-0\nline-1\nline-2\n");

      output.stdout = "";
      expect(await runGatewayCommand(["thor", "--head", "5", "--", "true"], runner, output, makeConfig(root))).toBe(0);
      expect(output.stdout).toContain("line-0");
      expect(output.stdout).not.toContain("line-7");
      expect(output.stderr).toBe("");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("run docker wraps the command in docker exec", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-run-docker-"));
    try {
      const runner = new FakeRunner();
      const output = { stdout: "", stderr: "" };
      const exitCode = await runGatewayCommand(["docker", "--", "bash", "-lc", "echo ok"], runner, output, makeConfig(root));

      expect(exitCode).toBe(0);
      expect(runner.calls.at(-1)?.args.join(" ")).toContain("docker exec pgx-lower-dev bash -lc");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test.each([
    ["thor", "--full", []],
    ["thor", "--head", ["payload"]],
    ["thor", "--tail", ["payload"]],
    ["docker", "--full", []],
    ["docker", "--head", ["payload"]],
    ["docker", "--tail", ["payload"]]
  ])("run %s preserves payload flag %s after --", async (target, flag, rest) => {
    const root = mkdtempSync(join(tmpdir(), "pgx-run-payload-"));
    try {
      const runner = new FakeRunner();
      const output = { stdout: "", stderr: "" };

      const exitCode = await runGatewayCommand([target, "--", "printf", "%s\\n", flag, ...rest], runner, output, makeConfig(root));

      expect(exitCode).toBe(0);
      const command = runner.calls.at(-1)?.args.join(" ") ?? "";
      expect(command).toContain(flag);
      for (const arg of rest) {
        expect(command).toContain(arg);
      }
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("run psql query uses postgres in the configured container", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-run-psql-"));
    try {
      const runner = new FakeRunner();
      const output = { stdout: "", stderr: "" };
      const exitCode = await runGatewayCommand(["psql", "--query", "SELECT 1"], runner, output, makeConfig(root));

      expect(exitCode).toBe(0);
      const command = runner.calls.at(-1)?.args.join(" ") ?? "";
      expect(command).toContain("docker exec pgx-lower-dev");
      expect(command).toContain("/usr/local/pgsql/bin/psql -v ON_ERROR_STOP=on -d regression");
      expect(command).toContain("SELECT 1");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  test("run psql file accepts only repo-local files and maps them to workspace", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-run-psql-file-"));
    try {
      mkdirSync(join(root, "tests", "debug"), { recursive: true });
      writeFileSync(join(root, "tests", "debug", "q17.sql"), "SELECT 1;\n");
      const runner = new FakeRunner();
      const output = { stdout: "", stderr: "" };

      const exitCode = await runGatewayCommand(["psql", "--file", join(root, "tests", "debug", "q17.sql")], runner, output, makeConfig(root));

      expect(exitCode).toBe(0);
      const command = runner.calls.at(-1)?.args.join(" ") ?? "";
      expect(command).toContain("chmod o+x /workspace /workspace/tests /workspace/tests/debug");
      expect(command).toContain("chmod o+r /workspace/tests/debug/q17.sql");
      expect(command).toContain("/workspace/tests/debug/q17.sql");

      const rejected = await runGatewayCommand(["psql", "--file", "/tmp/query.sql"], runner, output, makeConfig(root));
      expect(rejected).toBe(1);
      expect(output.stderr).toContain("outside the configured local checkout");

      writeFileSync(join(root, "tests", "debug", "q17.txt"), "SELECT 1;\n");
      const nonSql = await runGatewayCommand(["psql", "--file", join(root, "tests", "debug", "q17.txt")], runner, output, makeConfig(root));
      expect(nonSql).toBe(1);
      expect(output.stderr).toContain("expected a repo-local .sql file");

      const directory = await runGatewayCommand(["psql", "--file", join(root, "tests", "debug")], runner, output, makeConfig(root));
      expect(directory).toBe(1);
      expect(output.stderr).toContain("expected a repo-local .sql file");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
