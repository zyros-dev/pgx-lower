import { describe, expect, test } from "vitest";
import { summarizeFailure } from "../src/failure-summary.js";

describe("failure summary", () => {
  test.each([
    [
      "cmake",
      "dev-build-compile",
      "",
      "FAILED: extension/pgx_lower.so\n/workspace/src/a.cpp:42:13: error: no matching function\n",
      "compile-error"
    ],
    [
      "clang-tidy",
      "dev-lint-diff",
      "/workspace/src/a.cpp:10:5: warning: use auto [modernize-use-auto]\n",
      "",
      "clang-tidy"
    ],
    [
      "pg_regress",
      "dev-test-tpch",
      "not ok 3 - date 42 ms\ndiffs: /workspace/build-artifacts/ptest/extension/results/date.diff\n",
      "",
      "pg_regress"
    ],
    [
      "unit sql",
      "dev-test-unit-pg_nullability",
      "--- pg_nullability.sql ---\npsql:/workspace/tests/unit-tests/sql/pg_nullability.sql:19: ERROR: expected false\n",
      "",
      "unit-sql"
    ],
    [
      "psql",
      "run-psql",
      "psql:/tmp/q17.sql:19: ERROR: MLIR lowering pipeline failed\n",
      "",
      "psql"
    ],
    [
      "route",
      "route-check",
      "route assertion failed: id=tpch_q01 expected lower observed fallback\n",
      "",
      "route"
    ],
    ["mutagen", "sync-preflight", "alpha disconnected\n", "", "mutagen"],
    ["task spooler", "queue-status", "[job 12 queued on pgx-build]\nExit status: 1\n", "", "task-spooler"]
  ])("extracts %s evidence", (_name, commandName, stdout, stderr, kind) => {
    const summary = summarizeFailure({ commandName, stdoutSample: stdout, stderrSample: stderr, childExitCode: 1 });

    expect(summary?.kind).toBe(kind);
    expect(summary?.lines.length).toBeGreaterThan(0);
    expect(summary?.lines.length).toBeLessThanOrEqual(12);
  });

  test("uses postprocessed failures when the child exits zero", () => {
    const summary = summarizeFailure({
      commandName: "dev-test-tpch",
      stdoutSample: "1: ok 1 - smoke\n",
      stderrSample: "",
      childExitCode: 0,
      workflowExitCode: 1,
      postprocessedFailure: "REGRESSIONS: tpch_q01"
    });

    expect(summary?.kind).toBe("postprocessed");
    expect(summary?.lines.join("\n")).toContain("REGRESSIONS: tpch_q01");
  });

  test("prefers psql errors over task-spooler queue bookkeeping", () => {
    const summary = summarizeFailure({
      commandName: "dev-test-unit-pg_nullability",
      stdoutSample: [
        "[job 0 queued on pgx-build]",
        "psql:/workspace/tests/unit-tests/sql/pg_nullability.sql:9: ERROR: missing function"
      ].join("\n"),
      stderrSample: "",
      childExitCode: 3
    });

    expect(summary?.kind).toBe("unit-sql");
    expect(summary?.lines.join("\n")).toContain("missing function");
  });

  test("does not classify PostgreSQL problem diagnostics as mutagen failures", () => {
    const summary = summarizeFailure({
      commandName: "dev-test-unit-focused",
      stdoutSample: [
        "[job 0 queued on pgx-build]",
        "psql:/workspace/tests/unit-tests/sql/accepted_plan_verifier.sql:34: WARNING:  [PROBLEM:ERROR_LEVEL] verifier failed"
      ].join("\n"),
      stderrSample: "",
      childExitCode: 1
    });

    expect(summary?.kind).toBe("unit-sql");
    expect(summary?.lines.join("\n")).toContain("accepted_plan_verifier.sql");
  });

  test("omits unrelated unit SQL section headers when diagnostics are present", () => {
    const summary = summarizeFailure({
      commandName: "dev-test-unit-focused",
      stdoutSample: [
        "--- accepted_plan_verifier.sql ---",
        "psql:/workspace/tests/unit-tests/sql/accepted_plan_verifier.sql:34: WARNING:  [PROBLEM:ERROR_LEVEL] verifier failed",
        "--- numeric.sql ---",
        "--- pg_nullability.sql ---",
        "psql:/workspace/tests/unit-tests/sql/pg_nullability.sql:9: ERROR: missing function"
      ].join("\n"),
      stderrSample: "",
      childExitCode: 1
    });

    const rendered = summary?.lines.join("\n") ?? "";
    expect(summary?.kind).toBe("unit-sql");
    expect(rendered).toContain("accepted_plan_verifier.sql:34");
    expect(rendered).toContain("pg_nullability.sql:9");
    expect(rendered).not.toContain("--- numeric.sql ---");
  });

  test("omits warning-level PostgreSQL problem diagnostics from unit SQL failures", () => {
    const summary = summarizeFailure({
      commandName: "dev-test-unit-focused",
      stdoutSample: [
        "psql:/workspace/tests/unit-tests/sql/type_mapping.sql:40: WARNING:  [PROBLEM:WARNING_LEVEL] unsupported type",
        "psql:/workspace/tests/unit-tests/sql/pg_nullability.sql:9: ERROR: missing function"
      ].join("\n"),
      stderrSample: "",
      childExitCode: 1
    });

    const rendered = summary?.lines.join("\n") ?? "";
    expect(summary?.kind).toBe("unit-sql");
    expect(rendered).toContain("pg_nullability.sql:9");
    expect(rendered).not.toContain("WARNING_LEVEL");
  });

  test("classifies missing generated unit SQL suites as unit SQL failures", () => {
    const summary = summarizeFailure({
      commandName: "dev-test-unit-pg_nullability",
      stdoutSample: [
        "[job 0 queued on pgx-build]",
        "unit-sql: suite pg_nullability not generated under /workspace/tests/unit-tests/sql"
      ].join("\n"),
      stderrSample: "",
      childExitCode: 1
    });

    expect(summary?.kind).toBe("unit-sql");
    expect(summary?.lines.join("\n")).toContain("pg_nullability");
  });

  test("returns undefined when both child and workflow exit cleanly", () => {
    expect(
      summarizeFailure({ commandName: "ok", stdoutSample: "ok\n", stderrSample: "", childExitCode: 0, workflowExitCode: 0 })
    ).toBeUndefined();
  });
});
