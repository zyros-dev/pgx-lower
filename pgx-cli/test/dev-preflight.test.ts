import { mkdirSync, mkdtempSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { RunResult, StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { runDevPreflightCommand } from "../src/dev-preflight.js";
import type { DevConfig } from "../src/dev.js";

type FakeRunnerState = {
  localBranch?: string;
  localStatus?: string;
  localHead?: string;
  thorBranch?: string;
  thorStatus?: string;
  thorHead?: string;
  mutagenOk?: boolean;
  syncProofOk?: boolean;
};

class FakeRunner implements StreamingCommandRunner {
  readonly calls: Array<{ command: string; args: string[] }> = [];

  constructor(private readonly state: FakeRunnerState = {}) {}

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    const rendered = [command, ...args].join(" ");
    if (command === "git" && rendered.includes("branch --show-current")) {
      return ok(`${this.state.localBranch ?? "feature-a"}\n`);
    }
    if (command === "git" && rendered.includes("status --short")) {
      return ok(this.state.localStatus ?? "");
    }
    if (command === "git" && rendered.includes("rev-parse HEAD")) {
      return ok(`${this.state.localHead ?? "abc123"}\n`);
    }
    if (command === "ssh" && rendered.includes("git branch --show-current")) {
      return ok(`${this.state.thorBranch ?? "feature-a"}\n`);
    }
    if (command === "ssh" && rendered.includes("git status --short")) {
      return ok(this.state.thorStatus ?? "");
    }
    if (command === "ssh" && rendered.includes("git rev-parse HEAD")) {
      return ok(`${this.state.thorHead ?? "abc123"}\n`);
    }
    return ok("");
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args });
    const rendered = [command, ...args].join(" ");
    const stdout = rendered.includes("mutagen sync list")
      ? healthyJson(this.state.mutagenOk ?? true)
      : rendered.includes(".pgx-cli/sync-probes/dev-preflight.txt")
        ? (this.state.syncProofOk === false ? "wrong-proof\n" : "probe-dev-preflight\n")
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

function ok(stdout: string): RunResult {
  return { exitCode: 0, stdout, stderr: "" };
}

function healthyJson(ok: boolean): string {
  return JSON.stringify([
    {
      name: "pgx-lower",
      paused: false,
      status: ok ? "watching" : "halted",
      alpha: { connected: true },
      beta: { connected: true }
    }
  ]);
}

function makeConfig(options: {
  cliDistFresh?: boolean;
  unitSqlGenerated?: boolean;
  unitSqlFresh?: boolean;
  installGitHead?: string | null;
  installCommandName?: string;
  installCommand?: string[];
} = {}): DevConfig {
  const root = mkdtempSync(join(tmpdir(), "pgx-dev-preflight-test-"));
  mkdirSync(join(root, "pgx-cli", "src"), { recursive: true });
  mkdirSync(join(root, "pgx-cli", "dist"), { recursive: true });
  mkdirSync(join(root, "src", "pgx-lower", "test"), { recursive: true });
  mkdirSync(join(root, "tests", "unit-tests", "sql"), { recursive: true });
  mkdirSync(join(root, ".pgx-cli", "runs", "install"), { recursive: true });
  writeFileSync(join(root, "pgx-cli", "src", "index.ts"), "export {};\n");
  writeFileSync(join(root, "pgx-cli", "dist", "index.js"), "export {};\n");
  writeFileSync(join(root, "src", "pgx-lower", "test", "smoke_tests.cpp"), "PGX_TEST_FN(smoke) {}\n");
  const installSummary: Record<string, unknown> = {
    runId: "install",
    commandName: options.installCommandName ?? "dev-build-install-debug",
    command: options.installCommand ?? [
      "ssh",
      "comfy",
      "bash",
      "-c",
      "'export PATH=$HOME/.local/bin:$PATH && cd /home/zel/repos/pgx-lower && docker exec pgx-lower-dev cmake --install build-artifacts/ptest'"
    ],
    workflowExitCode: 0,
    finishedAt: "2099-01-01T00:00:00.000Z",
    artifactPaths: ["/workspace/build-artifacts/ptest"]
  };
  if (options.installGitHead !== null) {
    installSummary.gitHead = options.installGitHead ?? "abc123";
  }
  writeFileSync(join(root, ".pgx-cli", "runs", "install", "summary.json"), JSON.stringify(installSummary, null, 2) + "\n");
  if (options.unitSqlGenerated ?? true) {
    writeFileSync(join(root, "tests", "unit-tests", "sql", "smoke.sql"), "select 1;\n");
  }
  const oldDate = new Date("2026-01-01T00:00:00Z");
  const newDate = new Date("2026-01-02T00:00:00Z");
  if (options.cliDistFresh ?? true) {
    utimesSync(join(root, "pgx-cli", "src", "index.ts"), oldDate, oldDate);
    utimesSync(join(root, "pgx-cli", "dist", "index.js"), newDate, newDate);
  } else {
    utimesSync(join(root, "pgx-cli", "src", "index.ts"), newDate, newDate);
    utimesSync(join(root, "pgx-cli", "dist", "index.js"), oldDate, oldDate);
  }
  if (options.unitSqlGenerated ?? true) {
    const testSourceDate = options.unitSqlFresh === false ? newDate : oldDate;
    const sqlDate = options.unitSqlFresh === false ? oldDate : newDate;
    utimesSync(join(root, "src", "pgx-lower", "test", "smoke_tests.cpp"), testSourceDate, testSourceDate);
    utimesSync(join(root, "tests", "unit-tests", "sql", "smoke.sql"), sqlDate, sqlDate);
  }
  return {
    localProjectPath: root,
    remoteProjectPath: "/home/zel/repos/pgx-lower",
    sshHost: "comfy",
    mutagenSession: "pgx-lower",
    dockerContainer: "pgx-lower-dev",
    buildQueue: "pgx-build",
    checkQueue: "pgx-check",
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
      max_lines_total: 80,
      failure_tail_lines: 20,
      success_tail_lines: 10,
      progress: "final-summary",
      full_output_requires_flag: true
    }
  };
}

describe("dev preflight", () => {
  test("strict preflight fails when Mutagen sync proof does not match", async () => {
    const runner = new FakeRunner({ syncProofOk: false });
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict"], runner, output, makeConfig());

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("sync proof mismatch");
  });

  test("strict preflight anchors local git commands to configured project path", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = makeConfig();
    await runDevPreflightCommand(["--strict"], runner, output, config);

    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" "));
    expect(commands).toContain(`git -C ${config.localProjectPath} branch --show-current`);
    expect(commands).toContain(`git -C ${config.localProjectPath} status --short`);
    expect(commands).toContain(`git -C ${config.localProjectPath} rev-parse HEAD`);
  });

  test("strict preflight fails when thor branch differs from local branch", async () => {
    const runner = new FakeRunner({ localBranch: "feature-a", thorBranch: "main" });
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict"], runner, output, makeConfig());

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("thor branch differs");
  });

  test("strict preflight fails when thor HEAD differs from local HEAD", async () => {
    const runner = new FakeRunner({ localHead: "abc123", thorHead: "def456" });
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict"], runner, output, makeConfig());

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("thor HEAD differs");
  });

  test("strict preflight fails when thor worktree is dirty", async () => {
    const runner = new FakeRunner({ thorStatus: " M src/pgx-lower.cpp\n" });
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict"], runner, output, makeConfig());

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("thor worktree dirty");
  });

  test("strict preflight fails when pgx-cli dist is stale", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict"], runner, output, makeConfig({ cliDistFresh: false }));

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("pgx-cli dist stale");
  });

  test("strict preflight fails when generated unit SQL is missing", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict", "--unit-sql"], runner, output, makeConfig({ unitSqlGenerated: false }));

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("generated unit SQL missing");
  });

  test("strict preflight fails when an expected generated unit SQL suite is missing", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const config = makeConfig();
    writeFileSync(join(config.localProjectPath, "src", "pgx-lower", "test", "missing_suite_tests.cpp"), "PGX_TEST_FN(missing_suite) {}\n");
    const exitCode = await runDevPreflightCommand(["--strict", "--unit-sql"], runner, output, config);

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("generated unit SQL missing: missing_suite.sql");
  });

  test("strict preflight fails when generated unit SQL is older than its source", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict", "--unit-sql"], runner, output, makeConfig({ unitSqlFresh: false }));

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("generated unit SQL stale: smoke.sql");
  });

  test("strict preflight fails when install summary does not prove current HEAD", async () => {
    const runner = new FakeRunner({ localHead: "abc123", thorHead: "abc123" });
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict"], runner, output, makeConfig({ installGitHead: null }));

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("installed extension freshness not tied to current HEAD");
    expect(output.stderr).toContain("pgx-cli dev build install --profile debug");
  });

  test("strict preflight fails when install summary was built for another HEAD", async () => {
    const runner = new FakeRunner({ localHead: "abc123", thorHead: "abc123" });
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict"], runner, output, makeConfig({ installGitHead: "def456" }));

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("installed extension built for def456, current HEAD abc123");
  });

  test("strict preflight fails when latest summary is compile-only", async () => {
    const runner = new FakeRunner({ localHead: "abc123", thorHead: "abc123" });
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict"], runner, output, makeConfig({
      installCommandName: "dev-build-compile-debug",
      installCommand: [
        "ssh",
        "comfy",
        "bash",
        "-c",
        "'export PATH=$HOME/.local/bin:$PATH && cd /home/zel/repos/pgx-lower && docker exec pgx-lower-dev cmake --build build-artifacts/ptest'"
      ]
    }));

    expect(exitCode).toBe(1);
    expect(output.stderr).toContain("installed extension freshness not proven");
    expect(output.stderr).toContain("pgx-cli dev build install --profile debug");
  });

  test("strict preflight passes and prints a compact summary when state is clean", async () => {
    const runner = new FakeRunner();
    const output = { stdout: "", stderr: "" };
    const exitCode = await runDevPreflightCommand(["--strict"], runner, output, makeConfig());

    expect(exitCode).toBe(0);
    expect(output.stdout).toContain("dev preflight: ok");
    expect(output.stdout).toContain("- ok local branch");
    expect(output.stdout).toContain("- ok thor branch");
    expect(output.stdout).toContain("- ok pgx-cli dist");
    expect(output.stdout).toContain("- ok generated unit SQL");
    expect(output.stdout).toContain("- ok installed extension");
    expect(output.stderr).toBe("");
  });
});
