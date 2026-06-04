import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  classifyPsqlRegressionDelta,
  missingPsqlRegressionSourceMessage,
  parsePgRegressStatusLines,
  psqlRegressionDeltaExitCode,
  readPsqlRegressionBaseline,
  renderPsqlRegressionBaseline,
  renderPsqlRegressionSummary,
  validatePsqlRegressionSource
} from "../src/psql-regression-burndown.js";

describe("psql regression burndown helpers", () => {
  test("parses pg_regress passing and failing status lines", () => {
    const parsed = parsePgRegressStatusLines(
      [
        "ok 1 - boolean 10 ms",
        "1: not ok 3 - date 20 ms",
        "ignored diagnostic line"
      ].join("\n")
    );

    expect([...parsed.passing]).toEqual(["boolean"]);
    expect([...parsed.failing]).toEqual(["date"]);
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

  test("validates sql expected and schedule source paths", () => {
    const root = mkdtempSync(join(tmpdir(), "pgx-psql-source-"));
    mkdirSync(join(root, "sql"), { recursive: true });
    mkdirSync(join(root, "expected"), { recursive: true });
    writeFileSync(join(root, "parallel_schedule"), "test: boolean\n");

    expect(() => validatePsqlRegressionSource(root, (path) => path !== join(root, "expected"))).toThrow(
      /expected/
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
