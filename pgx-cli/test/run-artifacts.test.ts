import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  appendArtifactPath,
  createRunArtifactPaths,
  sanitizeRunName,
  writeCommand,
  writeRunSummary
} from "../src/run-artifacts.js";

function withTempRoot(testFn: (root: string) => void): void {
  const root = mkdtempSync(join(tmpdir(), "pgx-run-artifacts-"));
  try {
    testFn(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

describe("run artifacts", () => {
  test("creates stable run paths with sanitized command names", () => {
    withTempRoot((root) => {
      const paths = createRunArtifactPaths({
        root,
        commandName: "dev test/unit:pg_nullability!",
        now: new Date("2026-06-06T12:34:56.789Z"),
        randomSuffix: "a1b2c3"
      });

      expect(paths.runId).toBe("2026-06-06T12-34-56-789Z-dev-test-unit-pg-nullability-a1b2c3");
      expect(paths.runDir).toBe(join(root, ".pgx-cli", "runs", paths.runId));
      expect(existsSync(paths.runDir)).toBe(true);
      expect(paths.syncPreflightPath).toBe(join(paths.runDir, "sync-preflight.log"));
    });
  });

  test("writes command, summary, and artifact registry files", () => {
    withTempRoot((root) => {
      const paths = createRunArtifactPaths({ root, commandName: "dev-test", randomSuffix: "000001" });

      writeCommand(paths, ["pgx-cli", "dev", "test", "unit", "pg_nullability"]);
      writeRunSummary(paths, { runId: paths.runId, exitCode: 1 });
      appendArtifactPath(paths, "/tmp/pgx_ir/latest.mlir");

      expect(readFileSync(paths.commandPath, "utf8")).toBe("pgx-cli dev test unit pg_nullability\n");
      expect(JSON.parse(readFileSync(paths.summaryPath, "utf8"))).toEqual({
        runId: paths.runId,
        exitCode: 1
      });
      expect(readFileSync(paths.summaryPath, "utf8")).toMatch(/\n$/);
      expect(readFileSync(paths.artifactsPath, "utf8")).toBe("/tmp/pgx_ir/latest.mlir\n");
    });
  });

  test("honors configured transcript directory", () => {
    withTempRoot((root) => {
      const paths = createRunArtifactPaths({
        root,
        commandName: "dev-test",
        transcriptDir: "custom/runs",
        randomSuffix: "000001"
      });

      expect(paths.runDir).toBe(join(root, "custom", "runs", paths.runId));
      expect(existsSync(paths.runDir)).toBe(true);
    });
  });

  test("sanitizes punctuation to a compact run name", () => {
    expect(sanitizeRunName("a/b c_D.e:f!")).toBe("a-b-c-d-e-f");
  });
});
