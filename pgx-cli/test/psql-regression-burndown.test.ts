import { mkdirSync, mkdtempSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import type { CommandRunner, RunResult } from "../src/commands.js";
import {
  buildPsqlRegressionPgRegressCommand,
  classifyPsqlRegressionDelta,
  missingPsqlRegressionSourceMessage,
  parsePsqlRegressionBurndownArgs,
  parsePgRegressStatusLines,
  psqlRegressionDeltaExitCode,
  readPsqlRegressionBaseline,
  renderPsqlRegressionBaseline,
  renderPsqlRegressionSummary,
  runPsqlRegressionBurndownCommand,
  validatePsqlRegressionSource
} from "../src/psql-regression-burndown.js";

class FakeRunner implements CommandRunner {
  calls: Array<{ command: string; args: string[] }> = [];
  result: RunResult = { exitCode: 0, stdout: "ok 1 - boolean 10 ms\n", stderr: "" };

  async run(command: string, args: string[]): Promise<RunResult> {
    this.calls.push({ command, args });
    return this.result;
  }
}

describe("psql regression burndown helpers", () => {
  test("parses pg_regress passing and failing status lines", () => {
    const parsed = parsePgRegressStatusLines(
      [
        "ok 1 - boolean 10 ms",
        "1: not ok 3 - date 20 ms",
        "not ok 47 + opr_sanity 30 ms",
        "ignored diagnostic line"
      ].join("\n")
    );

    expect([...parsed.passing]).toEqual(["boolean"]);
    expect([...parsed.failing]).toEqual(["date", "opr_sanity"]);
  });

  test("classifies new failures, still failing, and newly passing baseline entries", () => {
    const delta = classifyPsqlRegressionDelta({
      passing: new Set(["boolean", "numeric"]),
      failing: new Set(["date", "join"]),
      baseline: new Set(["date", "numeric"])
    });

    expect([...delta.newFailures]).toEqual(["join"]);
    expect([...delta.stillFailing]).toEqual(["date"]);
    expect([...delta.nowPassing]).toEqual(["numeric"]);
  });

  test("fails on new failures and newly passing baseline entries", () => {
    const exactBaseline = classifyPsqlRegressionDelta({
      passing: new Set(["boolean"]),
      failing: new Set(["date"]),
      baseline: new Set(["date"])
    });
    const newFailure = classifyPsqlRegressionDelta({
      passing: new Set(["boolean"]),
      failing: new Set(["date", "join"]),
      baseline: new Set(["date"])
    });
    const nowPassing = classifyPsqlRegressionDelta({
      passing: new Set(["boolean", "date"]),
      failing: new Set<string>(),
      baseline: new Set(["date"])
    });

    expect(psqlRegressionDeltaExitCode(exactBaseline)).toBe(0);
    expect(psqlRegressionDeltaExitCode(newFailure)).toBe(1);
    expect(psqlRegressionDeltaExitCode(nowPassing)).toBe(1);
  });

  test("reads and writes baseline files with comments ignored", () => {
    const baseline = readPsqlRegressionBaseline("# comment\n\n date \njoin\n");

    expect([...baseline]).toEqual(["date", "join"]);
    expect(renderPsqlRegressionBaseline(baseline)).toBe("date\njoin\n");
  });

  test("reports concrete bootstrap instructions for missing upstream sources", () => {
    const message = missingPsqlRegressionSourceMessage("tests/psql-regression");

    expect(message).toContain("postgresql-17.6");
    expect(message).toContain("src/test/regress");
    expect(message).toContain("tests/psql-regression");
  });

  test("validates sql expected data resultmap and schedule source paths", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-psql-source-"));
    mkdirSync(join(root, "sql"), { recursive: true });
    mkdirSync(join(root, "expected"), { recursive: true });
    mkdirSync(join(root, "data"), { recursive: true });
    writeFileSync(join(root, "resultmap"), "");
    writeFileSync(join(root, "parallel_schedule"), "test: boolean\n");

    expect(() => validatePsqlRegressionSource(root, (path) => path !== join(root, "expected"))).toThrow(
      /expected/
    );
    expect(() => validatePsqlRegressionSource(root, (path) => path !== join(root, "data"))).toThrow(/data/);
    expect(() => validatePsqlRegressionSource(root, (path) => path !== join(root, "resultmap"))).toThrow(
      /resultmap/
    );
    expect(() => validatePsqlRegressionSource(root, () => true)).not.toThrow();
  });

  test("renders a markdown summary with counts names and route summary path", () => {
    const delta = classifyPsqlRegressionDelta({
      passing: new Set(["boolean", "numeric"]),
      failing: new Set(["date", "join"]),
      baseline: new Set(["date", "numeric"])
    });

    const summary = renderPsqlRegressionSummary({
      runName: "psql-regression-burndown",
      source: "tests/psql-regression",
      delta,
      routeSummaryPath: "build-artifacts/test-runs/psql-regression-burndown/route-summary.md"
    });

    expect(summary).toContain("# PostgreSQL Regression Burndown: psql-regression-burndown");
    expect(summary).toContain("Source: `tests/psql-regression`");
    expect(summary).toContain("Passing: 2");
    expect(summary).toContain("Failing: 2");
    expect(summary).toContain("Baseline known-failing: 2");
    expect(summary).toContain("join");
    expect(summary).toContain("numeric");
    expect(summary).toContain("route-summary.md");
  });
});

describe("psql regression burndown command", () => {
  test("builds a pg_regress command with source output schedule and extension paths", () => {
    const options = parsePsqlRegressionBurndownArgs([
      "--source",
      "tests/psql-regression",
      "--baseline",
      "baseline.txt",
      "--output-dir",
      "results",
      "--summary",
      "summary.md",
      "--route-summary",
      "route-summary.md",
      "--pg-regress",
      "pg_regress",
      "--bindir",
      "/usr/local/pgsql/bin",
      "--dlpath",
      "/usr/local/pgsql/lib",
      "--schedule",
      "parallel_schedule",
      "--load-extension",
      "pgx_lower"
    ]);

    expect(buildPsqlRegressionPgRegressCommand(options)).toEqual([
      "pg_regress",
      "--bindir=/usr/local/pgsql/bin",
      "--dlpath=/usr/local/pgsql/lib",
      "--inputdir=tests/psql-regression",
      "--outputdir=results",
      "--schedule=tests/psql-regression/parallel_schedule",
      "--load-extension=pgx_lower"
    ]);
  });

  test("defaults pg_regress to the installed pgxs path used by the dev container", () => {
    const options = parsePsqlRegressionBurndownArgs([]);

    expect(options.pgRegress).toBe("/usr/local/pgsql/lib/pgxs/src/test/regress/pg_regress");
  });

  test("record mode writes the current failing set as the reviewed baseline", async () => {
    const fixture = makeCommandFixture(["boolean", "date"]);
    const runner = new FakeRunner();
    runner.result = {
      exitCode: 1,
      stdout: "ok 1 - boolean 10 ms\nnot ok 2 - date 20 ms\n",
      stderr: ""
    };
    const io = { stdout: "", stderr: "" };

    const exitCode = await runPsqlRegressionBurndownCommand(
      [
        "--source",
        fixture.source,
        "--baseline",
        fixture.baseline,
        "--output-dir",
        fixture.outputDir,
        "--summary",
        fixture.summary,
        "--route-summary",
        fixture.routeSummary,
        "--record"
      ],
      runner,
      io
    );

    expect(exitCode).toBe(0);
    expect(readFileSync(fixture.baseline, "utf8")).toBe("date\n");
    expect(io.stdout).toContain(`Recorded 1 failing upstream PostgreSQL tests to ${fixture.baseline}`);
  });

  test("normal mode fails on newly failing upstream tests", async () => {
    const fixture = makeCommandFixture(["boolean", "date"]);
    writeFileSync(fixture.baseline, "");
    const runner = new FakeRunner();
    runner.result = {
      exitCode: 1,
      stdout: "ok 1 - boolean 10 ms\nnot ok 2 - date 20 ms\n",
      stderr: ""
    };
    const io = { stdout: "", stderr: "" };

    const exitCode = await runPsqlRegressionBurndownCommand(fixture.args, runner, io);

    expect(exitCode).toBe(1);
    expect(readFileSync(fixture.summary, "utf8")).toContain("date");
    expect(io.stderr).toContain("FAIL: PostgreSQL regression delta changed");
  });

  test("normal mode fails when a baseline entry starts passing", async () => {
    const fixture = makeCommandFixture(["boolean", "date"]);
    writeFileSync(fixture.baseline, "date\n");
    const runner = new FakeRunner();
    runner.result = {
      exitCode: 0,
      stdout: "ok 1 - boolean 10 ms\nok 2 - date 20 ms\n",
      stderr: ""
    };
    const io = { stdout: "", stderr: "" };

    const exitCode = await runPsqlRegressionBurndownCommand(fixture.args, runner, io);

    expect(exitCode).toBe(1);
    expect(readFileSync(fixture.summary, "utf8")).toContain("## Newly Passing\n- date");
  });

  test("no status transcript exits two with a log inspection message", async () => {
    const fixture = makeCommandFixture(["boolean"]);
    const runner = new FakeRunner();
    runner.result = { exitCode: 1, stdout: "cmake failed before tests ran\n", stderr: "" };
    const io = { stdout: "", stderr: "" };

    const exitCode = await runPsqlRegressionBurndownCommand(fixture.args, runner, io);

    expect(exitCode).toBe(2);
    expect(io.stderr).toContain("ERROR: no pg_regress status lines found");
    expect(io.stderr).toContain(join(fixture.outputDir, "pg_regress.log"));
  });

  test("route observation uses not_asserted defaults for upstream SQL without directives", async () => {
    const fixture = makeCommandFixture(["boolean"]);
    writeFileSync(fixture.baseline, "");
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runPsqlRegressionBurndownCommand(fixture.args, runner, io);

    expect(exitCode).toBe(0);
    expect(readFileSync(fixture.routeSummary, "utf8")).toContain("- Not asserted: 1");
    expect(io.stdout).toContain("OK: PostgreSQL regression delta matches baseline");
  });

  test("route observation ignores upstream SQL files with no pg_regress output", async () => {
    const fixture = makeCommandFixture(["boolean", "unused"]);
    unlinkSync(join(fixture.outputDir, "unused.out"));
    writeFileSync(fixture.baseline, "");
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runPsqlRegressionBurndownCommand(fixture.args, runner, io);

    expect(exitCode).toBe(0);
    expect(readFileSync(fixture.routeSummary, "utf8")).toContain("- Not asserted: 1");
    expect(io.stderr).not.toContain("INFO: route observation failed");
  });

  test("route observation reads pg_regress nested results directory", async () => {
    const fixture = makeCommandFixture(["boolean"]);
    unlinkSync(join(fixture.outputDir, "boolean.out"));
    mkdirSync(join(fixture.outputDir, "results"), { recursive: true });
    writeFileSync(join(fixture.outputDir, "results", "boolean.out"), "SELECT 1;\n");
    writeFileSync(fixture.baseline, "");
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runPsqlRegressionBurndownCommand(fixture.args, runner, io);

    expect(exitCode).toBe(0);
    expect(readFileSync(fixture.routeSummary, "utf8")).toContain("- Not asserted: 1");
  });

  test("writes pg_regress route check and summary artifacts", async () => {
    const fixture = makeCommandFixture(["boolean"]);
    writeFileSync(fixture.baseline, "");
    const runner = new FakeRunner();
    const io = { stdout: "", stderr: "" };

    const exitCode = await runPsqlRegressionBurndownCommand(fixture.args, runner, io);

    expect(exitCode).toBe(0);
    expect(readFileSync(join(fixture.outputDir, "pg_regress.log"), "utf8")).toContain("ok 1 - boolean");
    expect(readFileSync(join(fixture.outputDir, "route-check.log"), "utf8")).toContain("OK: route assertions passed");
    expect(readFileSync(fixture.summary, "utf8")).toContain("# PostgreSQL Regression Burndown");
    expect(readFileSync(fixture.routeSummary, "utf8")).toContain("# Route Summary");
  });
});

function makeCommandFixture(testNames: readonly string[]) {
  const root = mkdtempSync(join(tmpdir(), "pgx-psql-burndown-"));
  const source = join(root, "tests", "psql-regression");
  const outputDir = join(root, "results");
  const baseline = join(source, "baselines", "current.txt");
  const summary = join(root, "summary.md");
  const routeSummary = join(root, "route-summary.md");
  mkdirSync(join(source, "sql"), { recursive: true });
  mkdirSync(join(source, "expected"), { recursive: true });
  mkdirSync(join(source, "data"), { recursive: true });
  mkdirSync(join(source, "baselines"), { recursive: true });
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(join(source, "resultmap"), "");
  writeFileSync(join(source, "parallel_schedule"), `test: ${testNames.join(" ")}\n`);
  writeFileSync(baseline, "");

  for (const name of testNames) {
    writeFileSync(join(source, "sql", `${name}.sql`), "SELECT 1;\n");
    writeFileSync(join(source, "expected", `${name}.out`), "SELECT 1;\n");
    writeFileSync(join(outputDir, `${name}.out`), "SELECT 1;\n");
  }

  return {
    source,
    outputDir,
    baseline,
    summary,
    routeSummary,
    args: [
      "--source",
      source,
      "--baseline",
      baseline,
      "--output-dir",
      outputDir,
      "--summary",
      summary,
      "--route-summary",
      routeSummary
    ]
  };
}
