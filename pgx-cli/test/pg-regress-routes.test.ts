import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { RunResult, StreamingCommandRunner, StreamingRunOptions, StreamingRunResult } from "../src/commands.js";
import { parseRouteCheckArgs, runRouteCheckCommand } from "../src/pg-regress-routes.js";

class FakeRunner implements StreamingCommandRunner {
  calls: Array<{ command: string; args: string[]; streaming: boolean }> = [];
  result: RunResult = { exitCode: 0, stdout: "pg_regress ok\n", stderr: "" };

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args, streaming: false });
    return this.result;
  }

  async runStreaming(command: string, args: string[], options: StreamingRunOptions): Promise<StreamingRunResult> {
    this.calls.push({ command, args, streaming: true });
    options.stdout?.write(this.result.stdout);
    options.stderr?.write(this.result.stderr);
    return {
      childExitCode: this.result.exitCode,
      stdoutSample: { head: this.result.stdout, tail: "", truncated: false },
      stderrSample: { head: this.result.stderr, tail: "", truncated: false },
      timedOut: false
    };
  }
}

describe("route-check args", () => {
  test("parses required options and optional pg_regress command", () => {
    const options = parseRouteCheckArgs([
      "--run-name",
      "pgx",
      "--profile",
      "debug",
      "--execution-mode",
      "extension-auto",
      "--sql-dir",
      "tests/sql",
      "--output-dir",
      "results",
      "--summary",
      "summary.md",
      "--default-auto-should-route-to",
      "lower",
      "--require-route-directives",
      "--pg-regress",
      "--",
      "pg_regress",
      "--inputdir=tests"
    ]);

    expect(options.pgRegressCommand).toEqual(["pg_regress", "--inputdir=tests"]);
    expect(options.requireRouteDirectives).toBe(true);
  });
});

describe("route-check command", () => {
  test("writes a passing summary from existing pg_regress outputs", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-route-check-"));
    const sqlDir = join(root, "sql");
    const outputDir = join(root, "results");
    const summaryPath = join(root, "summary.md");
    mkdirSync(sqlDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });
    writeFileSync(
      join(sqlDir, "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q1 */\nSELECT 1;\n"
    );
    writeFileSync(join(outputDir, "queries.out"), "SELECT 1;\n ?column?\n----------\n        1\n");

    const io = { stdout: "", stderr: "" };
    const exitCode = await runRouteCheckCommand(
      [
        "--run-name",
        "pgx",
        "--profile",
        "debug",
        "--execution-mode",
        "extension-auto",
        "--sql-dir",
        sqlDir,
        "--output-dir",
        outputDir,
        "--summary",
        summaryPath,
        "--default-auto-should-route-to",
        "lower",
        "--require-route-directives"
      ],
      new FakeRunner(),
      io
    );

    expect(exitCode).toBe(0);
    expect(io.stdout).toContain("OK: route assertions passed");
    expect(readFileSync(summaryPath, "utf8")).toContain("# Route Summary: pgx");
  });

  test("returns one and writes a failing summary when route assertions fail", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-route-check-"));
    const sqlDir = join(root, "sql");
    const outputDir = join(root, "results");
    const summaryPath = join(root, "summary.md");
    mkdirSync(sqlDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });
    writeFileSync(
      join(sqlDir, "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=fallback id=q1 */\nSELECT 1;\n"
    );
    writeFileSync(join(outputDir, "queries.out"), "SELECT 1;\n ?column?\n----------\n        1\n");

    const io = { stdout: "", stderr: "" };
    const exitCode = await runRouteCheckCommand(
      [
        "--run-name",
        "pgx",
        "--profile",
        "debug",
        "--execution-mode",
        "extension-auto",
        "--sql-dir",
        sqlDir,
        "--output-dir",
        outputDir,
        "--summary",
        summaryPath,
        "--default-auto-should-route-to",
        "lower",
        "--require-route-directives"
      ],
      new FakeRunner(),
      io
    );

    expect(exitCode).toBe(1);
    expect(io.stderr).toContain("FAIL: route assertions failed");
    expect(readFileSync(summaryPath, "utf8")).toContain("q1");
  });

  test("runs supplied pg_regress command before inspecting outputs", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-route-check-"));
    const sqlDir = join(root, "sql");
    const outputDir = join(root, "results");
    mkdirSync(sqlDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });
    writeFileSync(join(sqlDir, "queries.sql"), "SELECT 1;\n");
    writeFileSync(join(outputDir, "queries.out"), "SELECT 1;\n");
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runRouteCheckCommand(
      [
        "--run-name",
        "pgx",
        "--profile",
        "debug",
        "--execution-mode",
        "extension-auto",
        "--sql-dir",
        sqlDir,
        "--output-dir",
        outputDir,
        "--summary",
        join(root, "summary.md"),
        "--default-auto-should-route-to",
        "not_asserted",
        "--pg-regress",
        "--",
        "pg_regress",
        "--inputdir=tests"
      ],
      runner,
      io
    );

    expect(exitCode).toBe(0);
    expect(runner.calls).toEqual([{ command: "pg_regress", args: ["--inputdir=tests"], streaming: true }]);
    expect(io.stdout).not.toContain("pg_regress ok");
    expect(readFileSync(join(outputDir, "pg_regress.log"), "utf8")).toContain("pg_regress ok");
  });

  test("preserves pg_regress failure when route assertions do not add failures", async () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-route-check-"));
    const sqlDir = join(root, "sql");
    const outputDir = join(root, "results");
    mkdirSync(sqlDir, { recursive: true });
    mkdirSync(outputDir, { recursive: true });
    writeFileSync(join(sqlDir, "queries.sql"), "SELECT 1;\n");
    writeFileSync(join(outputDir, "queries.out"), "SELECT 1;\n");
    const runner = new FakeRunner();
    runner.result = { exitCode: 2, stdout: "pg_regress failed\n", stderr: "" };
    const io = { stdout: "", stderr: "" };

    const exitCode = await runRouteCheckCommand(
      [
        "--run-name",
        "pgx",
        "--profile",
        "debug",
        "--execution-mode",
        "extension-auto",
        "--sql-dir",
        sqlDir,
        "--output-dir",
        outputDir,
        "--summary",
        join(root, "summary.md"),
        "--default-auto-should-route-to",
        "not_asserted",
        "--pg-regress",
        "--",
        "pg_regress"
      ],
      runner,
      io
    );

    expect(exitCode).toBe(2);
    expect(io.stderr).toContain("pg_regress exited 2");
  });
});
