import { relative, sep } from "node:path";
import { classifyComparisonMode } from "./compare-postgres-results.js";
import type { ComparisonMode } from "./compare-postgres-results.js";
import type { ComparePostgresTarget, SqlManifest, StatementManifestEntry } from "./sql-manifest.js";

export type RenderedComparePostgresQuery = {
  id: string;
  sourceFile: string;
  statementIndex: number;
  comparisonMode: ComparisonMode;
  routeExpectation: StatementManifestEntry["autoShouldRouteTo"];
};

export type NonComparableQuery = {
  id: string;
  sourceFile: string;
  statementIndex: number;
  reason: string;
};

export type RenderedComparePostgresFile = {
  sourceFile: string;
  stem: string;
  stockSql: string;
  extensionSql: string;
  queries: RenderedComparePostgresQuery[];
};

export type RenderedComparePostgresScripts = {
  setupFile?: string;
  files: RenderedComparePostgresFile[];
  comparableCount: number;
  nonComparableQueries: NonComparableQuery[];
};

const deterministicGucs = [
  "SET DateStyle = 'ISO, MDY';",
  "SET IntervalStyle = 'postgres';",
  "SET TimeZone = 'UTC';",
  "SET extra_float_digits = 3;",
  "SET client_min_messages = notice;"
];

export function renderComparePostgresScripts(options: {
  target: ComparePostgresTarget;
  manifests: readonly SqlManifest[];
}): RenderedComparePostgresScripts {
  if (options.target.executionMode !== "extension-auto") {
    throw new Error(`compare-postgres renderer cannot use ${options.target.executionMode} execution mode`);
  }
  const nonComparableQueries: NonComparableQuery[] = [];
  const files = options.manifests
    .map((manifest) => renderFile(options.target, manifest, nonComparableQueries))
    .filter((file) => file.queries.length > 0);
  const comparableCount = files.reduce((total, file) => total + file.queries.length, 0);

  return {
    ...(options.target.setupFile ? { setupFile: options.target.setupFile } : {}),
    files,
    comparableCount,
    nonComparableQueries
  };
}

function renderFile(
  target: ComparePostgresTarget,
  manifest: SqlManifest,
  nonComparableQueries: NonComparableQuery[]
): RenderedComparePostgresFile {
  if (target.executionMode === "stock") {
    throw new Error("compare-postgres renderer cannot use stock execution mode");
  }
  const stockParts = [...deterministicGucs];
  const extensionParts = [...deterministicGucs, "LOAD 'pgx_lower';", `SET pgx_lower.execution_mode = '${pgxLowerExecutionMode(target.executionMode)}';`];
  const queries: RenderedComparePostgresQuery[] = [];
  const lastComparableIndex = lastInterestingStatementIndex(manifest);

  for (const statement of manifest.statements) {
    if (statement.index > lastComparableIndex) {
      break;
    }

    if (!statement.comparable) {
      if (statement.statementKind !== "setup") {
        throw new Error(`${manifest.path}: statement ${statement.index} non-comparable statement before the last comparable query`);
      }
      appendSetup(statement, stockParts, extensionParts);
      continue;
    }

    const id = statement.id;
    if (!id) {
      throw new Error(`${manifest.path}: statement ${statement.index} requires an id`);
    }
    if (setupSqlReferencesPgxLower(statement.sql)) {
      throw new Error(`${manifest.path}: statement ${statement.index} comparable SQL cannot reference pgx_lower`);
    }
    const mode = classifyComparisonMode(statement.sql);
    if (mode === "non_comparable_nondeterministic_order") {
      nonComparableQueries.push({
        id,
        sourceFile: manifest.path,
        statementIndex: statement.index,
        reason: mode
      });
      continue;
    }
    stockParts.push(renderCopyBlock(id, mode, statement.sql));
    extensionParts.push(renderCopyBlock(id, mode, statement.sql, ["SET pgx_lower.execution_mode = 'auto';"]));
    queries.push({
      id,
      sourceFile: manifest.path,
      statementIndex: statement.index,
      comparisonMode: mode,
      routeExpectation: statement.autoShouldRouteTo
    });
  }

  return {
    sourceFile: manifest.path,
    stem: relative(target.sqlDir, manifest.path).split(sep).join("/"),
    stockSql: `${stockParts.join("\n\n")}\n`,
    extensionSql: `${extensionParts.join("\n\n")}\n`,
    queries
  };
}

function lastInterestingStatementIndex(manifest: SqlManifest): number {
  const comparable = manifest.statements.filter((statement) => statement.comparable);
  return comparable.at(-1)?.index ?? -1;
}

function appendSetup(statement: StatementManifestEntry, stockParts: string[], extensionParts: string[]): void {
  if (statement.statementKind !== "setup") {
    return;
  }
  if (setupSqlUsesPsqlMetaCommand(statement.sql)) {
    throw new Error(`${statement.path}: statement ${statement.index} setup SQL cannot use psql meta commands`);
  }
  if (isPgxLowerSetup(statement.sql)) {
    return;
  }
  if (setupSqlUsesLoad(statement.sql)) {
    throw new Error(`${statement.path}: statement ${statement.index} setup SQL cannot use LOAD in compare-postgres`);
  }
  if (setupSqlUsesDoBlock(statement.sql)) {
    throw new Error(`${statement.path}: statement ${statement.index} setup SQL cannot use DO blocks in compare-postgres`);
  }
  if (setupSqlReferencesPgxLower(statement.sql)) {
    throw new Error(`${statement.path}: statement ${statement.index} setup SQL cannot reference pgx_lower`);
  }
  if (setupSqlConfiguresPreloadLibraries(statement.sql)) {
    throw new Error(`${statement.path}: statement ${statement.index} setup SQL cannot configure preload libraries`);
  }
  stockParts.push(statement.sql);
  extensionParts.push(statement.sql);
}

function isPgxLowerSetup(sql: string): boolean {
  const stripped = stripLeadingSqlComments(sql);
  return /^\s*LOAD\s+['"]pgx_lower(?:\.so)?['"]\s*;?\s*$/iu.test(stripSqlComments(stripped)) || /^\s*SET\s+pgx_lower\./iu.test(stripped);
}

export function setupSqlReferencesPgxLower(sql: string): boolean {
  return decodePostgresEscapes(stripSqlComments(sql)).toLowerCase().replace(/[^a-z0-9]+/gu, "").includes("pgxlower");
}

export function setupSqlUsesDoBlock(sql: string): boolean {
  return /^\s*DO\b/iu.test(stripLeadingSqlComments(sql));
}

export function setupSqlUsesPsqlMetaCommand(sql: string): boolean {
  return stripLeadingSqlComments(sql).trimStart().startsWith("\\");
}

export function setupSqlUsesLoad(sql: string): boolean {
  return /^\s*LOAD\b/iu.test(stripLeadingSqlComments(sql));
}

export function setupSqlConfiguresPreloadLibraries(sql: string): boolean {
  return /\b(?:session|shared|local)_preload_libraries\b/iu.test(stripSqlComments(sql));
}

function pgxLowerExecutionMode(mode: ComparePostgresTarget["executionMode"]): string {
  if (mode === "extension-auto") return "auto";
  if (mode === "force-fallback") return "force_fallback";
  if (mode === "force-lower") return "force_lower";
  return "auto";
}

function stripLeadingSqlComments(sql: string): string {
  let i = 0;
  while (i < sql.length) {
    while (/\s/.test(sql[i] ?? "")) {
      i++;
    }
    if (sql[i] === "-" && sql[i + 1] === "-") {
      const end = sql.indexOf("\n", i + 2);
      i = end === -1 ? sql.length : end + 1;
      continue;
    }
    if (sql[i] === "/" && sql[i + 1] === "*") {
      const end = sql.indexOf("*/", i + 2);
      if (end === -1) {
        return sql.slice(i);
      }
      i = end + 2;
      continue;
    }
    break;
  }
  return sql.slice(i);
}

function stripSqlComments(sql: string): string {
  return sql.replace(/--[^\n]*(?:\n|$)/gu, "\n").replace(/\/\*[\s\S]*?\*\//gu, " ");
}

function decodePostgresEscapes(sql: string): string {
  return sql
    .replace(/\\\+([0-9A-Fa-f]{6})/gu, (_match, hex: string) => codePoint(hex))
    .replace(/\\([0-9A-Fa-f]{4})/gu, (_match, hex: string) => codePoint(hex))
    .replace(/\\x([0-9A-Fa-f]{1,2})/gu, (_match, hex: string) => codePoint(hex))
    .replace(/\\([0-7]{1,3})/gu, (_match, octal: string) => codePoint(Number.parseInt(octal, 8).toString(16)))
    .replace(/\\(.)/gu, "$1");
}

function codePoint(hex: string): string {
  const value = Number.parseInt(hex, 16);
  return Number.isFinite(value) ? String.fromCodePoint(value) : "";
}

function renderCopyBlock(id: string, mode: ComparisonMode, sql: string, extraSetup: readonly string[] = []): string {
  return [
    ...deterministicGucs,
    ...extraSetup,
    `\\echo __PGX_COMPARE_START__ id=${id} order=${mode}`,
    "COPY (",
    stripTrailingSemicolon(sql),
    ") TO STDOUT WITH (FORMAT csv, HEADER true, NULL '\\N');",
    `\\echo __PGX_COMPARE_END__ id=${id}`
  ].join("\n");
}

function stripTrailingSemicolon(sql: string): string {
  return sql.trim().replace(/;\s*$/u, "");
}
