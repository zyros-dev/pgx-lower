import { describe, expect, test } from "vitest";
import { recordGateFailure, shouldBlockReviewGate } from "../src/gate-memory.js";

describe("gate memory", () => {
  test("records focused reproducer for a failed review step", () => {
    const state = recordGateFailure({
      gate: "review",
      head: "abc123",
      stepName: "utest-pg",
      stepCommand: ["pgx-cli", "dev", "test", "focused"],
      runId: "run-1"
    });

    expect(state.focusedCommand).toEqual(["pgx-cli", "dev", "test", "focused"]);
  });

  test("blocks blind rerun on same head without focused evidence", () => {
    const decision = shouldBlockReviewGate({
      strict: true,
      currentHead: "abc123",
      previousFailure: {
        gate: "review",
        head: "abc123",
        stepName: "utest-pg",
        focusedCommand: ["pgx-cli", "dev", "test", "focused"],
        runId: "run-1"
      },
      argv: ["gate", "review"]
    });

    expect(decision.blocked).toBe(true);
    expect(decision.message).toContain("run focused reproducer first");
  });
});
