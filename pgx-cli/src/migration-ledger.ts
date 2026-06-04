import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parse } from "yaml";

export type MigrationOutcome = "wrap" | "move" | "keep-internal" | "delete";

export type MigrationClassification = {
  path: string;
  kind: "script" | "recipe" | "recipe-file" | "generated";
  outcome: MigrationOutcome;
  replacement: string;
  reason: string;
};

export type MigrationLedger = {
  version: 1;
  classifications: MigrationClassification[];
};

const validOutcomes = new Set(["wrap", "move", "keep-internal", "delete"]);
const validKinds = new Set(["script", "recipe", "recipe-file", "generated"]);

export function loadMigrationLedger(root: string = process.cwd()): MigrationLedger {
  const path = join(root, "pgx-cli-migration-ledger.yaml");
  return parse(readFileSync(path, "utf8")) as MigrationLedger;
}

export function validateMigrationLedger(root: string, ledger: MigrationLedger): string[] {
  const errors: string[] = [];
  if (ledger.version !== 1) {
    errors.push(`invalid ledger version ${String(ledger.version)}`);
  }
  for (const entry of ledger.classifications ?? []) {
    if (!validOutcomes.has(entry.outcome)) {
      errors.push(`${entry.path}: invalid outcome ${entry.outcome}`);
    }
    if (!validKinds.has(entry.kind)) {
      errors.push(`${entry.path}: invalid kind ${entry.kind}`);
    }
    if (!entry.path || !entry.reason || !entry.replacement) {
      errors.push(`${entry.path}: path, replacement, and reason are required`);
    }
    if (entry.kind !== "recipe" && !existsSync(join(root, entry.path))) {
      errors.push(`${entry.path}: path does not exist`);
    }
  }
  return errors;
}

export function formatLedgerSummary(ledger: MigrationLedger): string {
  const counts = new Map<MigrationOutcome, number>();
  for (const entry of ledger.classifications) {
    counts.set(entry.outcome, (counts.get(entry.outcome) ?? 0) + 1);
  }
  return [
    `ledger entries: ${ledger.classifications.length}`,
    `wrap: ${counts.get("wrap") ?? 0}`,
    `move: ${counts.get("move") ?? 0}`,
    `keep-internal: ${counts.get("keep-internal") ?? 0}`,
    `delete: ${counts.get("delete") ?? 0}`
  ].join("\n") + "\n";
}
