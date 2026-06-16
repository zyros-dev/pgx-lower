import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { RunResult, StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import {
  parseComparePostgresArgs,
  runComparePostgresCliCommand,
  runComparePostgresInternalCommand
} from "../src/compare-postgres.js";
import { DEFAULT_OUTPUT_CONFIG, DEFAULT_SYNC_CONFIG } from "../src/config.js";

class FakeRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];
  queryOutputs = new Map<string, RunResult>();

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    if (command === "mutagen" && args[1] === "list") {
      return { exitCode: 0, stdout: healthyJson(), stderr: "" };
    }
    const rendered = [command, ...args].join(" ");
    for (const [needle, result] of this.queryOutputs) {
      if (rendered.includes(needle)) {
        return result;
      }
    }
    return { exitCode: 0, stdout: "", stderr: "" };
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args });
    const rendered = [command, ...args].join(" ");
    const stdout = command === "mutagen" && args[1] === "list"
      ? healthyJson()
      : rendered.includes(".pgx-cli/sync-probes/")
      ? `probe-${rendered.match(new RegExp("sync-probes/([^/']+)\\.txt"))?.[1] ?? "missing"}\n`
      : "remote compare ok\n";
    options.stdout?.write(stdout);
    return {
      childExitCode: 0,
      stdoutSample: { head: stdout, tail: "", truncated: false },
      stderrSample: { head: "", tail: "", truncated: false },
      timedOut: false
    };
  }
}

function healthyJson(): string {
  return JSON.stringify([{ name: "pgx-lower", paused: false, status: "watching", alpha: { connected: true }, beta: { connected: true } }]);
}

function makeRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "pgx-compare-command-"));
  mkdirSync(join(root, "tests", "sql"), { recursive: true });
  writeFileSync(join(root, "tests", "sql", "setup.sql"), "CREATE TABLE t(id int);\nINSERT INTO t VALUES (1), (2);\n");
  writeFileSync(
    join(root, "tests", "sql", "queries.sql"),
    "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q1 */\nSELECT * FROM t ORDER BY id;\n"
  );
  writeFileSync(
    join(root, "tests", "workloads.yaml"),
    `
workloads:
  tiny:
    sql: tests/sql
    setup: tests/sql/setup.sql
    compare_postgres:
      expected_comparable_count: 1
tests:
  tiny-correctness:
    workload: tiny
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
  );
  return root;
}

const config = {
  mutagenSession: "pgx-lower",
  sshHost: "comfy",
  remoteProjectPath: "/home/zel/repos/pgx-lower",
  dockerContainer: "pgx-lower-dev",
  sync: DEFAULT_SYNC_CONFIG,
  output: DEFAULT_OUTPUT_CONFIG
};

function makeConfig() {
  return {
    ...config,
    localProjectPath: mkdtempSync(join(tmpdir(), "pgx-compare-config-"))
  };
}

async function withInternalGuard<T>(callback: () => Promise<T>): Promise<T> {
  const previous = process.env.PGX_COMPARE_POSTGRES_INTERNAL;
  process.env.PGX_COMPARE_POSTGRES_INTERNAL = "1";
  try {
    return await callback();
  } finally {
    if (previous === undefined) {
      delete process.env.PGX_COMPARE_POSTGRES_INTERNAL;
    } else {
      process.env.PGX_COMPARE_POSTGRES_INTERNAL = previous;
    }
  }
}

async function runInternal(
  args: readonly string[],
  runner: StreamingCommandRunner,
  io: { stdout: string; stderr: string }
): Promise<number> {
  return withInternalGuard(() => runComparePostgresInternalCommand(args, runner, io, { allowUnsafeLocalForTests: true }));
}

describe("compare-postgres args", () => {
  test("parses workload and artifact options", () => {
    const options = parseComparePostgresArgs([
      "--workload",
      "tpch-correctness",
      "--root",
      "/workspace",
      "--run-name",
      "manual",
      "--output-dir",
      "/tmp/out"
    ]);

    expect(options.workload).toBe("tpch-correctness");
    expect(options.root).toBe("/workspace");
    expect(options.runName).toBe("manual");
    expect(options.outputDir).toBe("/tmp/out");
  });

  test("requires workload", () => {
    expect(() => parseComparePostgresArgs([])).toThrow(/Usage: pgx-cli test compare-postgres/);
  });

  test("defaults root to the nearest repo root", () => {
    const root = makeRoot();
    mkdirSync(join(root, "pgx-cli"), { recursive: true });
    const previous = process.cwd();
    process.chdir(join(root, "pgx-cli"));
    try {
      expect(parseComparePostgresArgs(["--workload", "tiny-correctness"]).root).toBe(realpathSync(root));
    } finally {
      process.chdir(previous);
    }
  });
});

describe("compare-postgres internal command", () => {
  test("rejects the private entrypoint without the managed-runner environment guard", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runComparePostgresInternalCommand(
      ["--workload", "tiny-correctness", "--root", root, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("requires PGX_COMPARE_POSTGRES_INTERNAL=1");
    expect(runner.calls).toEqual([]);
  });

  test("rejects the private entrypoint outside the container even with the environment guard", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await withInternalGuard(() =>
      runComparePostgresInternalCommand(
        ["--workload", "tiny-correctness", "--root", root, "--from-managed-runner"],
        runner,
        io
      )
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("requires the /workspace container environment");
    expect(runner.calls).toEqual([]);
  });

  test("writes scripts, outputs, JSON diff, and Markdown summary for a match", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n2\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n2\n__PGX_COMPARE_END__ id=q1\n",
      stderr: "NOTICE:  [PGX-LOWER] [ROUTE:NOTICE] fallback unsupported_plan: fixture\n"
    });
    const outputDir = join(root, "artifacts", "compare-postgres");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(0);
    expect(io.stdout).toContain("OK: compare-postgres passed");
    expect(existsSync(join(outputDir, "scripts", "stock", "queries.sql"))).toBe(true);
    expect(existsSync(join(outputDir, "outputs", "extension", "queries.stderr"))).toBe(true);
    const summary = readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8");
    const diff = readFileSync(join(outputDir, "compare-postgres-diff.json"), "utf8");
    expect(summary).toContain("Compared queries: 1");
    expect(summary).toContain("fallback unsupported_plan");
    expect(diff).toContain("\"workload\": \"tiny-correctness\"");
    expect(diff).toContain("fallback unsupported_plan");
    expect(JSON.parse(diff).comparisons).toEqual([
      expect.objectContaining({
        queryId: "q1",
        sourceFile: join(root, "tests", "sql", "queries.sql"),
        statementIndex: 0,
        comparisonMode: "ordered",
        routeExpectation: "lower",
        stockRowCount: 2,
        extensionRowCount: 2,
        mismatchKind: null,
        preview: null
      })
    ]);
  });

  test("fails when query output contains unexpected result blocks", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: [
        "__PGX_COMPARE_START__ id=q1 order=ordered",
        "id",
        "1",
        "__PGX_COMPARE_END__ id=q1",
        "__PGX_COMPARE_START__ id=unexpected order=ordered",
        "id",
        "stock-only",
        "__PGX_COMPARE_END__ id=unexpected"
      ].join("\n"),
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: [
        "__PGX_COMPARE_START__ id=q1 order=ordered",
        "id",
        "1",
        "__PGX_COMPARE_END__ id=q1",
        "__PGX_COMPARE_START__ id=unexpected order=ordered",
        "id",
        "extension-only",
        "__PGX_COMPARE_END__ id=unexpected"
      ].join("\n"),
      stderr: ""
    });
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("unexpected_result_block");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("unexpected_result_block");
  });

  test("does not use su when the internal command is already non-root", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n2\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n2\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });

    await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      { stdout: "", stderr: "" }
    );

    expect(runner.calls.some((call) => call.command === "su")).toBe(false);
  });

  test("returns nonzero and keeps bounded stdout on mismatch", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n2\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("FAIL: compare-postgres mismatch");
    expect(`${io.stdout}${io.stderr}`).not.toContain("__PGX_COMPARE_START__");
  });

  test("default runs isolate scratch databases and artifact directories", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).not.toContain("pgx_compare_tiny_correctness_stock");
    expect(io.stdout).not.toContain("test-runs/tiny_correctness/compare-postgres");
    expect(io.stdout).toMatch(/test-runs\/20[0-9]{6}-[0-9]{6}-[a-f0-9]{6}-tiny_correctness\/compare-postgres/);
  });

  test("two default runs in the same second use different run names", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const first = { stdout: "", stderr: "" };
    const second = { stdout: "", stderr: "" };

    expect(await runInternal(["--workload", "tiny-correctness", "--root", root, "--from-managed-runner"], runner, first)).toBe(0);
    expect(await runInternal(["--workload", "tiny-correctness", "--root", root, "--from-managed-runner"], runner, second)).toBe(0);

    const firstSummary = first.stdout.match(/Summary: (.*compare-postgres-summary\.md)/)?.[1];
    const secondSummary = second.stdout.match(/Summary: (.*compare-postgres-summary\.md)/)?.[1];
    expect(firstSummary).toBeTruthy();
    expect(secondSummary).toBeTruthy();
    expect(firstSummary).not.toBe(secondSummary);
  });

  test("sanitizes run-name before using it in default artifact paths", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--run-name", "../../tmp/x", "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(0);
    expect(io.stdout).toContain("build-artifacts/test-runs/tmp_x/compare-postgres/compare-postgres-summary.md");
    expect(io.stdout).not.toContain("../");
  });

  test("long run names still produce distinct stock and extension database names", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--run-name", "a".repeat(120), "--from-managed-runner"],
      runner,
      { stdout: "", stderr: "" }
    );

    expect(exitCode).toBe(0);
    const created = runner.calls
      .map((call) => [call.command, ...call.args].join(" "))
      .filter((command) => command.includes("createdb"))
      .map((command) => command.match(/createdb ([A-Za-z0-9_]+)/)?.[1])
      .filter((name): name is string => !!name);
    expect(created).toHaveLength(2);
    expect(new Set(created).size).toBe(2);
    expect(created.every((name) => name.length <= 63)).toBe(true);
  });

  test("database reset failures stop before query execution and write artifacts", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("dropdb --if-exists", {
      exitCode: 1,
      stdout: "",
      stderr: "database is being accessed by other users\n"
    });
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("database_reset_failed");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("database_reset_failed");
    expect(readFileSync(join(outputDir, "compare-postgres-diff.json"), "utf8")).toContain("database is being accessed");
    expect(runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n")).not.toContain("psql -v ON_ERROR_STOP");
  });

  test("cleans up scratch databases after query execution", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      { stdout: "", stderr: "" }
    );

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" "));
    const dropCommands = commands.filter((command) => command.includes("dropdb --if-exists"));
    expect(dropCommands).toHaveLength(4);
    expect(commands.findLastIndex((command) => command.includes("dropdb --if-exists"))).toBeGreaterThan(
      commands.findLastIndex((command) => command.includes("extension/queries.sql"))
    );
  });

  test("psql invocations skip startup files and clear PGOPTIONS", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      { stdout: "", stderr: "" }
    );

    expect(exitCode).toBe(0);
    const psqlCommands = runner.calls
      .map((call) => [call.command, ...call.args].join(" "))
      .filter((command) => command.includes("/usr/local/pgsql/bin/psql"));
    expect(psqlCommands.length).toBeGreaterThan(0);
    expect(psqlCommands.every((command) => command.includes("env -u PGOPTIONS /usr/local/pgsql/bin/psql -X -v ON_ERROR_STOP=on"))).toBe(true);
  });

  test("scratch_database false is rejected when the workload has setup SQL", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tiny:
    sql: tests/sql
    setup: tests/sql/setup.sql
    compare_postgres:
      expected_comparable_count: 1
      scratch_database: false
tests:
  tiny-correctness:
    workload: tiny
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("scratch_database: false cannot be used with workload setup SQL");
    expect(runner.calls).toEqual([]);
  });

  test("rejects pgx-lower setup in workload setup SQL", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "setup.sql"),
      [
        "LOAD 'pgx_lower.so';",
        "SET pgx_lower.execution_mode = 'force_lower';",
        "CREATE TABLE t AS SELECT 1 AS id;"
      ].join("\n")
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("workload setup SQL cannot load or configure pgx_lower");
    expect(runner.calls).toEqual([]);
  });

  test("rejects hidden pgx-lower references in workload setup SQL", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "setup.sql"),
      [
        "ALTER DATABASE postgres SET pgx_lower.execution_mode = 'force_lower';",
        "CREATE TABLE t AS SELECT 1 AS id;"
      ].join("\n")
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("workload setup SQL cannot load or configure pgx_lower");
    expect(runner.calls).toEqual([]);
  });

  test("rejects psql include commands in workload setup SQL", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "hidden.sql"),
      "LOAD 'pgx_lower';\nSET pgx_lower.execution_mode = 'auto';\n"
    );
    writeFileSync(
      join(root, "tests", "sql", "setup.sql"),
      "\\i tests/sql/hidden.sql\nCREATE TABLE t AS SELECT 1 AS id;\n"
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("workload setup SQL cannot use psql meta commands");
    expect(runner.calls).toEqual([]);
  });

  test("rejects dynamically assembled pgx-lower setup through DO blocks", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "setup.sql"),
      [
        "DO $$",
        "BEGIN",
        "  EXECUTE 'LOAD ' || 'pgx_' || 'lower';",
        "END $$;",
        "CREATE TABLE t AS SELECT 1 AS id;"
      ].join("\n")
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("workload setup SQL cannot use DO blocks");
    expect(runner.calls).toEqual([]);
  });

  test("rejects DO blocks in workload setup SQL", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "setup.sql"),
      [
        "DO $$",
        "BEGIN",
        "  EXECUTE 'LOAD ' || quote_literal(chr(112)||chr(103)||chr(120)||chr(95)||chr(108)||chr(111)||chr(119)||chr(101)||chr(114));",
        "END $$;",
        "CREATE TABLE t AS SELECT 1 AS id;"
      ].join("\n")
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("workload setup SQL cannot use DO blocks");
    expect(runner.calls).toEqual([]);
  });

  test("rejects obfuscated pgx-lower LOAD in workload setup SQL", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "setup.sql"),
      [
        String.raw`LOAD E'pgx\137lower.so';`,
        "CREATE TABLE t AS SELECT 1 AS id;"
      ].join("\n")
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("workload setup SQL cannot load or configure pgx_lower");
    expect(runner.calls).toEqual([]);
  });

  test("rejects preload library changes in workload setup SQL", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "setup.sql"),
      [
        "ALTER ROLE postgres SET session_preload_libraries = 'auto_explain';",
        "CREATE TABLE t AS SELECT 1 AS id;"
      ].join("\n")
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("workload setup SQL cannot configure preload libraries");
    expect(runner.calls).toEqual([]);
  });

  test("rejects escaped pgx-lower references in workload setup SQL", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "setup.sql"),
      [
        String.raw`CREATE SCHEMA U&"pgx\005flower";`,
        "CREATE TABLE t AS SELECT 1 AS id;"
      ].join("\n")
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("workload setup SQL cannot load or configure pgx_lower");
    expect(runner.calls).toEqual([]);
  });

  test("rejects row-producing statements in workload setup SQL", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "setup.sql"),
      [
        "CREATE TABLE t AS SELECT 1 AS id;",
        "SELECT * FROM t;"
      ].join("\n")
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("workload setup SQL cannot contain row-producing statements");
    expect(runner.calls).toEqual([]);
  });

  test("fails when stock run emits pgx-lower diagnostics", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: "NOTICE:  [PGX-LOWER] [ROUTE:NOTICE] fallback impossible on stock\n"
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: "NOTICE:  [PGX-LOWER] [ROUTE:NOTICE] fallback allowed on extension\n"
    });
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("stock_pgx_lower_diagnostic");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("stock_pgx_lower_diagnostic");
  });

  test("fails when stock setup emits pgx-lower diagnostics", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set(`_stock -f ${join(root, "tests", "sql", "setup.sql")}`, {
      exitCode: 0,
      stdout: "",
      stderr: "NOTICE:  [PGX-LOWER] [ROUTE:NOTICE] stock setup contaminated\n"
    });
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("stock_pgx_lower_diagnostic");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("stock_pgx_lower_diagnostic");
  });

  test("fails for any pgx-lower diagnostic on the stock side", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: "WARNING:  [PGX-LOWER] [PROBLEM:WARNING] contaminated stock\n"
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("stock_pgx_lower_diagnostic");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("stock_pgx_lower_diagnostic");
  });

  test("fails when extension run emits force-fallback diagnostics", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: "NOTICE:  [PGX-LOWER] [ROUTE:NOTICE] fallback force_fallback: execution mode forced stock PostgreSQL\n"
    });
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("extension_forced_fallback_diagnostic");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("extension_forced_fallback_diagnostic");
  });

  test("scratch_database false runs no-setup workloads against postgres without reset and records the mode", async () => {
    const root = makeRoot();
    unlinkSync(join(root, "tests", "sql", "setup.sql"));
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tiny:
    sql: tests/sql
    compare_postgres:
      expected_comparable_count: 1
      scratch_database: false
tests:
  tiny-correctness:
    workload: tiny
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const outputDir = join(root, "artifacts");

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      { stdout: "", stderr: "" }
    );

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).not.toContain("dropdb");
    expect(commands).not.toContain("createdb");
    expect(commands).toContain("-d postgres");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("Scratch databases: disabled");
  });

  test("scratch_database false rejects setup-only SQL files before comparable queries", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tiny:
    sql: tests/sql
    compare_postgres:
      expected_comparable_count: 1
      scratch_database: false
tests:
  tiny-correctness:
    workload: tiny
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("setup.sql: source file has no comparable statements");
    expect(runner.calls).toEqual([]);
  });

  test("scratch_database false rejects inline setup SQL before comparable queries", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "queries.sql"),
      [
        "DELETE FROM t;",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q1 */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tiny:
    sql: tests/sql
    compare_postgres:
      expected_comparable_count: 1
      scratch_database: false
      exclude_files:
        - setup.sql
      exclude_reasons:
        setup.sql: setup-only fixture for this command test
tests:
  tiny-correctness:
    workload: tiny
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("scratch_database: false cannot be used with inline setup SQL");
    expect(runner.calls).toEqual([]);
  });

  test("expected comparable count mismatch writes non-comparable query artifacts", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_limit */\nSELECT * FROM t LIMIT 1;\n"
    );
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      new FakeRunner(),
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("expected_comparable_count_mismatch");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("non_comparable_nondeterministic_order");
    expect(readFileSync(join(outputDir, "compare-postgres-diff.json"), "utf8")).toContain("q_limit");
  });

  test("fails when an ID-bearing TABLE statement has no result block", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "queries.sql"),
      [
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q1 */",
        "SELECT * FROM t ORDER BY id;",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_table */",
        "TABLE t;"
      ].join("\n")
    );
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tiny:
    sql: tests/sql
    setup: tests/sql/setup.sql
tests:
  tiny-correctness:
    workload: tiny
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n2\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n2\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("missing_result_block");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("q_table");
  });

  test("Markdown summary lists concrete excluded files with reasons", async () => {
    const root = makeRoot();
    writeFileSync(join(root, "tests", "sql", "skip_a.sql"), "SELECT 2;\n");
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tiny:
    sql: tests/sql
    setup: tests/sql/setup.sql
    compare_postgres:
      expected_comparable_count: 1
      exclude_files:
        - skip_*.sql
      exclude_reasons:
        skip_*.sql: unsupported fixture
tests:
  tiny-correctness:
    workload: tiny
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const outputDir = join(root, "artifacts");

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      { stdout: "", stderr: "" }
    );

    expect(exitCode).toBe(0);
    const summary = readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8");
    expect(summary).toContain("skip_a.sql: unsupported fixture");
    expect(summary).toContain("q1");
  });

  test("fails instead of passing an empty workload", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-compare-empty-"));
    mkdirSync(join(root, "tests", "empty"), { recursive: true });
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  empty:
    sql: tests/empty
tests:
  empty-correctness:
    workload: empty
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
`
    );
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "empty-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      new FakeRunner(),
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("no_comparable_queries");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("no_comparable_queries");
  });

  test("DML returning is a fatal manifest error without run artifacts", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_delete */\nDELETE FROM t RETURNING *;\n"
    );
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      new FakeRunner(),
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("non_comparable_dml_returning");
    expect(existsSync(join(outputDir, "compare-postgres-summary.md"))).toBe(false);
  });

  test("data-modifying CTE returning is a fatal manifest error without run artifacts", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_delete */\nWITH moved AS (DELETE FROM t RETURNING *) SELECT * FROM moved;\n"
    );
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      new FakeRunner(),
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("non_comparable_dml_returning");
    expect(existsSync(join(outputDir, "compare-postgres-summary.md"))).toBe(false);
  });

  test("data-modifying CTE without returning is a fatal manifest error without run artifacts", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "sql", "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_delete */\nWITH deleted AS (DELETE FROM t) SELECT 1;\n"
    );
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      new FakeRunner(),
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("non_comparable_data_modifying_cte");
    expect(existsSync(join(outputDir, "compare-postgres-summary.md"))).toBe(false);
  });

  test("writes summary and diff when marker parsing fails", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const outputDir = join(root, "artifacts");
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", outputDir, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("stock_result_parse_failed");
    expect(readFileSync(join(outputDir, "compare-postgres-summary.md"), "utf8")).toContain("stock_result_parse_failed");
    expect(readFileSync(join(outputDir, "compare-postgres-diff.json"), "utf8")).toContain("missing end marker");
    expect(`${io.stdout}${io.stderr}`).not.toContain("__PGX_COMPARE_START__");
  });

  test("unknown workloads list valid workload and test names", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "missing", "--root", root, "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("unknown compare-postgres workload missing");
    expect(io.stderr).toContain("tiny-correctness");
    expect(io.stderr).toContain("tiny");
  });
});

describe("compare-postgres managed public command", () => {
  test("returns nonzero usage for missing workload but zero for help", async () => {
    const runner = new FakeRunner();
    const missing = { stdout: "", stderr: "" };
    const help = { stdout: "", stderr: "" };

    expect(await runComparePostgresCliCommand([], runner, missing, makeConfig())).toBe(1);
    expect(missing.stderr).toContain("Usage: pgx-cli test compare-postgres --workload <name>");
    expect(await runComparePostgresCliCommand(["--help"], runner, help, makeConfig())).toBe(0);
    expect(help.stdout).toContain("Usage: pgx-cli test compare-postgres --workload <name>");
  });

  test("rejects unknown options instead of dispatching remotely", async () => {
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runComparePostgresCliCommand(
      ["--workload", "tpch-correctness", "--bogus", "x"],
      runner,
      io,
      makeConfig()
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("Unknown option --bogus");
    expect(runner.calls).toEqual([]);
  });

  test("local invocation dispatches remotely instead of invoking local PostgreSQL tools", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };
    const exitCode = await runComparePostgresCliCommand(
      ["--workload", "tiny-correctness", "--root", root],
      runner,
      io,
      makeConfig()
    );

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("ssh comfy");
    expect(commands).toContain("test compare-postgres-internal");
    expect(commands).toContain("PGX_COMPARE_POSTGRES_INTERNAL=1");
    expect(commands).toContain("npm --prefix pgx-cli install && npm --prefix pgx-cli run build");
    expect(commands).toContain("cmake --install .");
    expect(commands).toContain("/workspace/build-artifacts/compare-postgres/");
    expect(commands).not.toContain("cd /workspace/build-artifacts/ptest");
    expect(commands).toContain("CTestTestfile.cmake");
    expect(commands).toContain("/workspace/src/lingodb/mlir");
    expect(commands.indexOf("cmake --install .")).toBeLessThan(commands.indexOf("compare-postgres-internal"));
    expect(commands).toContain("chmod -R o+rX /workspace");
    expect(commands).not.toContain("[ -d pgx-cli/node_modules ]");
    expect(commands).not.toMatch(/(?:^|\s)(psql|createdb|dropdb)(?:\s|$)/);
    expect(io.stdout).toContain("transcript:");
  });

  test("forwards public output-dir to the managed internal command", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runComparePostgresCliCommand(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", "build-artifacts/custom-compare"],
      runner,
      io,
      makeConfig()
    );

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("--output-dir /workspace/build-artifacts/custom-compare");
    expect(commands).not.toContain("/workspace/build-artifacts/test-runs/tpch_correctness/compare-postgres");
  });

  test("uses the workload profile for the managed extension build", async () => {
    const root = makeRoot();
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tiny:
    sql: tests/sql
    setup: tests/sql/setup.sql
    compare_postgres:
      expected_comparable_count: 1
tests:
  tiny-correctness:
    workload: tiny
    profile: release
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const runner = new FakeRunner();

    const exitCode = await runComparePostgresCliCommand(
      ["--workload", "tiny-correctness", "--root", root],
      runner,
      { stdout: "", stderr: "" },
      makeConfig()
    );

    expect(exitCode).toBe(0);
    const commands = runner.calls.map((call) => [call.command, ...call.args].join(" ")).join("\n");
    expect(commands).toContain("-DCMAKE_BUILD_TYPE=Release");
    expect(commands).toContain("/workspace/build-artifacts/compare-postgres/");
    expect(commands).toContain("/release");
  });

  test("fails clearly when workload setup SQL fails", async () => {
    const root = makeRoot();
    const runner = new FakeRunner();
    runner.queryOutputs.set("setup.sql", {
      exitCode: 1,
      stdout: "",
      stderr: "psql: error: setup.sql: Permission denied\n"
    });
    runner.queryOutputs.set("stock/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n2\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    runner.queryOutputs.set("extension/queries.sql", {
      exitCode: 0,
      stdout: "__PGX_COMPARE_START__ id=q1 order=ordered\nid\n1\n2\n__PGX_COMPARE_END__ id=q1\n",
      stderr: ""
    });
    const io = { stdout: "", stderr: "" };

    const exitCode = await runInternal(
      ["--workload", "tiny-correctness", "--root", root, "--output-dir", join(root, "artifacts"), "--from-managed-runner"],
      runner,
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("setup_run_failed");
    expect(existsSync(join(root, "artifacts", "scripts", "stock", "queries.sql"))).toBe(true);
    expect(existsSync(join(root, "artifacts", "scripts", "extension", "queries.sql"))).toBe(true);
  });
});
