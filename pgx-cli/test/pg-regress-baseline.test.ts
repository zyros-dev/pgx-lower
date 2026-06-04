import { describe, expect, test } from "vitest";
import { evaluatePgRegressBaseline } from "../src/pg-regress-baseline.js";

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
});
