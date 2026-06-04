import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { loadMigrationLedger, validateMigrationLedger } from "../src/migration-ledger.js";

function withRepo(testFn: (root: string) => void): void {
  const root = mkdtempSync(join(tmpdir(), "pgx-ledger-"));
  try {
    testFn(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

describe("migration ledger", () => {
  test("loads valid classifications", () => {
    withRepo((root) => {
      mkdirSync(join(root, "scripts"), { recursive: true });
      writeFileSync(join(root, "scripts", "run_lint.sh"), "");
      writeFileSync(
        join(root, "pgx-cli-migration-ledger.yaml"),
        `
version: 1
classifications:
  - path: scripts/run_lint.sh
    kind: script
    outcome: wrap
    replacement: pgx-cli dev lint diff
    reason: agents should call pgx-cli
`
      );

      const ledger = loadMigrationLedger(root);
      expect(ledger.classifications[0].path).toBe("scripts/run_lint.sh");
      expect(validateMigrationLedger(root, ledger)).toEqual([]);
    });
  });

  test("rejects invalid outcome", () => {
    withRepo((root) => {
      writeFileSync(
        join(root, "pgx-cli-migration-ledger.yaml"),
        `
version: 1
classifications:
  - path: missing.sh
    kind: script
    outcome: maybe
    replacement: pgx-cli debug missing
    reason: bad outcome
`
      );

      const ledger = loadMigrationLedger(root);
      expect(validateMigrationLedger(root, ledger)).toContain("missing.sh: invalid outcome maybe");
    });
  });
});
