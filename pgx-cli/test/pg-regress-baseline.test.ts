import { describe, expect, test } from "vitest";
import { detectCtestFailure, evaluatePgRegressBaseline, summarizePgRegressBaseline } from "../src/pg-regress-baseline.js";

describe("pg_regress baseline evaluation", () => {
  test("all passing with no baseline exits zero", () => {
    const result = evaluatePgRegressBaseline("1: ok 1 - 1_one_tuple 10 ms\n", "");

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("OK: no new regressions vs baseline.");
  });

  test("known failure in baseline exits zero", () => {
    const result = evaluatePgRegressBaseline("1: not ok 2 - 2_known_red 50 ms\n", "2_known_red\n");

    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("OK: no new regressions vs baseline.");
  });

  test("new failure exits one", () => {
    const result = evaluatePgRegressBaseline("1: not ok 2 - 2_new_regression 50 ms\n", "");

    expect(result.exitCode).toBe(1);
    expect(result.stdout).toContain("REGRESSIONS");
  });

  test("bail out emits explicit red marker", () => {
    const result = evaluatePgRegressBaseline(
      [
        "1: ok 1 - 1_one_tuple 10 ms",
        "1: diff: /workspace/tests/expected/43_version.out: No such file or directory",
        "1: Bail out!diff command failed"
      ].join("\n"),
      ""
    );

    expect(result.exitCode).toBe(1);
    expect(result.stdout).toContain("NEW TEST NEEDS EXPECTED FILE");
    expect(result.stdout).toContain("43_version");
    expect(result.stdout).not.toContain("OK: no new regressions vs baseline.");
  });

  test("input without TAP-ish lines exits two", () => {
    const result = evaluatePgRegressBaseline("cmake failed before tests ran\n", "");

    expect(result.exitCode).toBe(2);
    expect(result.stderr).toContain("ERROR: no TAP-ish lines found");
  });

  test("summary helper returns failed tests and workflow exit code", () => {
    const result = summarizePgRegressBaseline(
      [
        "1: ok 1 - 1_one_tuple 10 ms",
        "1: not ok 2 - 2_new_regression 50 ms",
        "diffs: /workspace/build-artifacts/ptest/extension/results/2_new_regression.diff"
      ].join("\n"),
      ""
    );

    expect(result.workflowExitCode).toBe(1);
    expect(result.lines).toContain("failed test: 2_new_regression");
    expect(result.lines).toContain("diff: /workspace/build-artifacts/ptest/extension/results/2_new_regression.diff");
    expect(result.lines.join("\n")).toContain("new regression");
  });

  test("detects CTest failures when the shell command captured output with child exit zero", () => {
    const result = detectCtestFailure(
      [
        "40% tests passed, 3 tests failed out of 5",
        "",
        "The following tests FAILED:",
        "\t  2 - pgx_lower_regress_routes (Failed)",
        "\t  5 - pgx_lower_tpch_routes (Failed)",
        "Errors while running CTest"
      ].join("\n")
    );

    expect(result).toContain("CTest failed");
    expect(result).toContain("pgx_lower_regress_routes");
    expect(result).toContain("pgx_lower_tpch_routes");
  });

  test("does not flag a zero-failure CTest summary", () => {
    expect(detectCtestFailure("100% tests passed, 0 tests failed out of 5\n")).toBeUndefined();
  });
});
