import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { basename, dirname, isAbsolute, join, relative, sep } from "node:path";
import { parse } from "yaml";

export const validRoutes = ["lower", "fallback", "ignore", "not_asserted"] as const;
export type RouteExpectation = (typeof validRoutes)[number];

export type RouteExecutionMode = "stock" | "extension-auto" | "force-fallback" | "force-lower";
export type StatementKind = "query" | "setup" | "other";
export type NonComparableReason = "non_comparable_dml_returning" | "non_comparable_data_modifying_cte";

export type StatementDirective = {
  autoShouldRouteTo: RouteExpectation;
  id: string | undefined;
};

export type StatementManifestEntry = {
  path: string;
  index: number;
  sql: string;
  autoShouldRouteTo: RouteExpectation;
  id: string | undefined;
  statementKind: StatementKind;
  comparable: boolean;
  nonComparableReason?: NonComparableReason;
};

export type SqlManifest = {
  path: string;
  statements: StatementManifestEntry[];
};

export type ComparePostgresExcludedFile = {
  glob: string;
  reason: string;
  matchedFiles: string[];
};

export type ComparePostgresTarget = {
  name: string;
  workloadName: string;
  root: string;
  sqlDir: string;
  setupFile?: string;
  profile: string;
  executionMode: RouteExecutionMode;
  defaultAutoShouldRouteTo: RouteExpectation;
  requireRouteDirectives: boolean;
  scratchDatabase: boolean;
  excludedFiles: ComparePostgresExcludedFile[];
  expectedComparableCount?: number;
};

export class RouteConfigError extends Error {}

const directiveRe = /^\/\*\s*<<pgx-lower-config>>:\s*(.*?)\s*\*\/$/s;
const keyValueRe = /^([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)$/;
const markerSafeIdRe = /^[A-Za-z0-9_.:-]+$/;

export function parseSqlManifest(options: {
  sql: string;
  path: string;
  defaultRoute: RouteExpectation;
  requireRouteDirectives: boolean;
}): SqlManifest {
  const statements = splitSqlStatements(options.sql, options.path);
  let pending: StatementDirective | undefined;
  let statementIndex = 0;
  const manifestStatements: StatementManifestEntry[] = [];

  for (const item of statements) {
    if (item.kind === "directive") {
      if (pending) {
        throw new RouteConfigError(`${options.path}: route directive is not attached to a statement`);
      }
      pending = parseDirective(item.text, options.path);
      continue;
    }

    const sql = item.text.trim();
    if (!sql) {
      continue;
    }

    const classification = classifyStatement(sql);
    if (options.requireRouteDirectives && classification.statementKind === "query" && !pending) {
      throw new RouteConfigError(`${options.path}: statement ${statementIndex} missing route directive`);
    }

    const route = pending?.autoShouldRouteTo ?? (classification.statementKind === "setup" ? "ignore" : options.defaultRoute);
    if (options.requireRouteDirectives && classification.statementKind === "query" && !pending?.id) {
      throw new RouteConfigError(`${options.path}: statement ${statementIndex} requires an id`);
    }

    manifestStatements.push({
      path: options.path,
      index: statementIndex,
      sql,
      autoShouldRouteTo: route,
      id: pending?.id,
      statementKind: classification.statementKind,
      comparable: classification.comparable,
      ...(classification.nonComparableReason ? { nonComparableReason: classification.nonComparableReason } : {})
    });
    pending = undefined;
    statementIndex++;
  }

  if (pending) {
    throw new RouteConfigError(`${options.path}: route directive is not attached to a following statement`);
  }

  return { path: options.path, statements: manifestStatements };
}

export function validateGlobalIds(manifests: readonly SqlManifest[]): void {
  const seen = new Map<string, StatementManifestEntry>();
  for (const manifest of manifests) {
    for (const statement of manifest.statements) {
      if (!statement.id) {
        continue;
      }
      const existing = seen.get(statement.id);
      if (existing) {
        throw new RouteConfigError(
          `duplicate route directive id ${statement.id}: ${existing.path}:${existing.index} and ${statement.path}:${statement.index}`
        );
      }
      seen.set(statement.id, statement);
    }
  }
}

export function resolveComparePostgresTarget(options: {
  root: string;
  workload: string;
  workloadsPath?: string;
}): ComparePostgresTarget {
  const root = options.root;
  const workloadsPath = options.workloadsPath ?? join(root, "tests", "workloads.yaml");
  const config = readYamlObject(workloadsPath);
  const tests = objectValue(config.tests, "tests", false);
  const workloads = objectValue(config.workloads, "workloads", true);
  const testEntry = objectValue(tests?.[options.workload], `tests.${options.workload}`, false);

  let workloadName = options.workload;
  let testConfig: Record<string, unknown> | undefined;
  if (testEntry) {
    testConfig = testEntry;
    workloadName = stringValue(testEntry.workload, `tests.${options.workload}.workload`);
  }

  const workload = objectValue(
    workloads[workloadName],
    `workloads.${workloadName}`,
    true,
    () => unknownWorkloadMessage(options.workload, tests, workloads)
  );
  const targetName = testConfig ? options.workload : workloadName;
  const compareConfig = {
    ...objectValue(workload.compare_postgres, `workloads.${workloadName}.compare_postgres`, false),
    ...objectValue(testConfig?.compare_postgres, `tests.${targetName}.compare_postgres`, false)
  };
  const sqlDir = join(root, stringValue(workload.sql, `workloads.${workloadName}.sql`));
  const setup = typeof workload.setup === "string" ? join(root, workload.setup) : undefined;
  const excludedFiles = resolveExcludedFiles(sqlDir, compareConfig);
  const scratchDatabase = typeof compareConfig.scratch_database === "boolean" ? compareConfig.scratch_database : true;
  if (setup && !scratchDatabase) {
    throw new RouteConfigError("scratch_database: false cannot be used with workload setup SQL");
  }

  const executionMode = executionModeValue(testConfig ? testConfig.execution_mode ?? workload.execution_mode ?? "extension-auto" : "extension-auto");
  if (executionMode !== "extension-auto") {
    throw new RouteConfigError("compare-postgres execution_mode must be extension-auto");
  }

  return {
    name: targetName,
    workloadName,
    root,
    sqlDir,
    ...(setup ? { setupFile: setup } : {}),
    profile: typeof testConfig?.profile === "string" ? testConfig.profile : "debug",
    executionMode,
    defaultAutoShouldRouteTo: routeValue(
      testConfig ? testConfig.default_auto_should_route_to ?? workload.default_auto_should_route_to ?? "not_asserted" : "not_asserted"
    ),
    requireRouteDirectives: typeof testConfig?.require_route_directives === "boolean" ? testConfig.require_route_directives : false,
    scratchDatabase,
    excludedFiles,
    ...(typeof compareConfig.expected_comparable_count === "number"
      ? { expectedComparableCount: compareConfig.expected_comparable_count }
      : {})
  };
}

export function readComparePostgresManifests(target: ComparePostgresTarget): SqlManifest[] {
  const excluded = new Set(target.excludedFiles.flatMap((item) => item.matchedFiles));
  const setup = target.setupFile ? normalizeAbsolute(target.setupFile) : undefined;
  const manifests = compareSourceFiles(target, excluded, setup)
    .map((path) =>
      parseSqlManifest({
        path,
        sql: readFileSync(path, "utf8"),
        defaultRoute: target.defaultAutoShouldRouteTo,
        requireRouteDirectives: target.requireRouteDirectives
      })
    );

  validateGlobalIds(manifests);
  validatePinnedComparePostgresManifests(target, manifests);
  for (const manifest of manifests) {
    for (const statement of manifest.statements) {
      if (statement.nonComparableReason) {
        throw new RouteConfigError(`${manifest.path}: statement ${statement.index} ${statement.nonComparableReason}`);
      }
      if (statement.comparable && !statement.id) {
        throw new RouteConfigError(`${manifest.path}: statement ${statement.index} requires an id`);
      }
    }
  }
  validateNoDroppedNonComparableFiles(manifests);
  return manifests;
}

function compareSourceFiles(target: ComparePostgresTarget, excluded: Set<string>, setup: string | undefined): string[] {
  const discovered = sqlFiles(target.sqlDir)
    .filter((path) => normalizeAbsolute(path) !== setup)
    .filter((path) => !excluded.has(relativePosix(target.sqlDir, path)));
  const pinned = pinnedComparePostgresSourceFiles(target);
  if (!pinned) {
    return discovered;
  }

  const byRelative = new Map(discovered.map((path) => [relativePosix(target.sqlDir, path), path]));
  const unexpected = [...byRelative.keys()].filter((path) => !pinned.includes(path));
  if (unexpected.length > 0) {
    throw new RouteConfigError(`unexpected compare-postgres source file ${unexpected[0]} for ${target.name}`);
  }
  const missing = pinned.filter((path) => !byRelative.has(path));
  if (missing.length > 0) {
    throw new RouteConfigError(`missing pinned compare-postgres source file ${missing[0]} for ${target.name}`);
  }
  const manifests = pinned.map((path) => byRelative.get(path)!);
  return manifests;
}

function pinnedComparePostgresSourceFiles(target: ComparePostgresTarget): string[] | undefined {
  if (target.name === "tpch-correctness" && target.workloadName === "tpch") {
    return ["tpch.sql", "tpch_no_lower.sql"];
  }
  return undefined;
}

function validatePinnedComparePostgresManifests(target: ComparePostgresTarget, manifests: readonly SqlManifest[]): void {
  const pinned = pinnedComparePostgresSourceFiles(target);
  if (!pinned) {
    return;
  }
  const expectedPerFile = target.name === "tpch-correctness" && target.workloadName === "tpch" ? 22 : undefined;
  if (expectedPerFile === undefined) {
    return;
  }
  const byRelative = new Map(manifests.map((manifest) => [relativePosix(target.sqlDir, manifest.path), manifest]));
  for (const file of pinned) {
    const actual = byRelative.get(file)?.statements.filter((statement) => statement.comparable).length ?? 0;
    if (actual !== expectedPerFile) {
      throw new RouteConfigError(`${file} expected ${expectedPerFile} comparable statements, found ${actual}`);
    }
  }
}

function validateNoDroppedNonComparableFiles(manifests: readonly SqlManifest[]): void {
  for (const manifest of manifests) {
    if (manifest.statements.some((statement) => statement.comparable)) {
      continue;
    }
    const nonSetup = manifest.statements.find((statement) => statement.statementKind !== "setup");
    if (nonSetup) {
      throw new RouteConfigError(`${manifest.path}: non-comparable statement ${nonSetup.index} must be excluded`);
    }
    throw new RouteConfigError(`${manifest.path}: source file has no comparable statements; exclude it with a reason`);
  }
}

export function isQueryStatement(sql: string): boolean {
  return classifyStatement(sql).statementKind === "query";
}

type SplitItem = { kind: "directive" | "statement"; text: string };

function splitSqlStatements(sql: string, path: string): SplitItem[] {
  const items: SplitItem[] = [];
  let current = "";
  let i = 0;

  while (i < sql.length) {
    const ch = sql[i];
    const next = sql[i + 1];

    if (ch === "\\" && sqlPrefixIsOnlyWhitespaceAndComments(current)) {
      const end = sql.indexOf("\n", i);
      const statement = (end === -1 ? sql.slice(i) : sql.slice(i, end)).trim();
      if (statement) {
        items.push({ kind: "statement", text: statement });
      }
      current = "";
      i = end === -1 ? sql.length : end + 1;
      continue;
    }

    if (ch === "'") {
      const [text, end] = readSingleQuoted(sql, i, isEscapeStringQuote(sql, i));
      current += text;
      i = end;
      continue;
    }

    if (ch === '"') {
      const [text, end] = readDoubleQuoted(sql, i);
      current += text;
      i = end;
      continue;
    }

    if (ch === "$") {
      const dollar = tryReadDollarQuoted(sql, i);
      if (dollar) {
        current += dollar.text;
        i = dollar.end;
        continue;
      }
    }

    if (ch === "-" && next === "-") {
      const end = sql.indexOf("\n", i + 2);
      const comment = end === -1 ? sql.slice(i) : sql.slice(i, end + 1);
      current += comment;
      i += comment.length;
      continue;
    }

    if (ch === "/" && next === "*") {
      const { text, end } = readBlockComment(sql, i, path);
      if (text.includes("pgx-lower-config") && !directiveRe.test(text.trim())) {
        throw new RouteConfigError(`${path}: malformed pgx-lower-config directive`);
      }
      const directiveMatch = directiveRe.exec(text.trim());
      if (directiveMatch) {
        if (!sqlPrefixIsOnlyWhitespaceAndComments(current)) {
          throw new RouteConfigError(`${path}: route directive appears in the middle of a statement`);
        }
        current = "";
        items.push({ kind: "directive", text: text.trim() });
      } else {
        current += text;
      }
      i = end;
      continue;
    }

    current += ch;
    i++;

    if (ch === ";") {
      const statement = current.trim();
      if (statement) {
        items.push({ kind: "statement", text: statement });
      }
      current = "";
    }
  }

  if (current.trim() && !sqlPrefixIsOnlyWhitespaceAndComments(current)) {
    items.push({ kind: "statement", text: current.trim() });
  }

  return items;
}

function sqlPrefixIsOnlyWhitespaceAndComments(text: string): boolean {
  let i = 0;
  while (i < text.length) {
    if (/\s/.test(text[i] ?? "")) {
      i++;
      continue;
    }
    if (text[i] === "-" && text[i + 1] === "-") {
      const end = text.indexOf("\n", i + 2);
      i = end === -1 ? text.length : end + 1;
      continue;
    }
    if (text[i] === "/" && text[i + 1] === "*") {
      const end = text.indexOf("*/", i + 2);
      if (end === -1) {
        return false;
      }
      i = end + 2;
      continue;
    }
    return false;
  }
  return true;
}

function parseDirective(text: string, path: string): StatementDirective {
  const match = directiveRe.exec(text);
  const body = match?.[1];
  if (body === undefined) {
    throw new RouteConfigError(`${path}: malformed route directive`);
  }

  const directive: Partial<StatementDirective> = {};
  const seen = new Set<string>();
  for (const token of body.trim().split(/\s+/).filter((item) => item.length > 0)) {
    const kv = keyValueRe.exec(token);
    if (!kv) {
      throw new RouteConfigError(`${path}: malformed route directive token ${token}`);
    }
    const key = kv[1];
    const value = kv[2];
    if (!key || !value) {
      throw new RouteConfigError(`${path}: malformed route directive token ${token}`);
    }
    if (seen.has(key)) {
      throw new RouteConfigError(`${path}: duplicate route directive key ${key}`);
    }
    seen.add(key);

    if (key === "auto_should_route_to") {
      if (!isRouteExpectation(value)) {
        throw new RouteConfigError(`${path}: invalid auto_should_route_to value ${value}`);
      }
      directive.autoShouldRouteTo = value;
    } else if (key === "id") {
      if (!markerSafeIdRe.test(value)) {
        throw new RouteConfigError(`${path}: invalid route directive id ${value}`);
      }
      directive.id = value;
    } else {
      throw new RouteConfigError(`${path}: unknown route directive key ${key}`);
    }
  }

  if (!directive.autoShouldRouteTo) {
    throw new RouteConfigError(`${path}: route directive missing auto_should_route_to`);
  }
  return { autoShouldRouteTo: directive.autoShouldRouteTo, id: directive.id };
}

function classifyStatement(sql: string): {
  statementKind: StatementKind;
  comparable: boolean;
  nonComparableReason?: NonComparableReason;
} {
  const keyword = firstKeyword(sql);
  const nonComparableReason = nonComparableDmlReason(sql);
  if (nonComparableReason) {
    return { statementKind: "other", comparable: false, nonComparableReason };
  }
  if (keyword === "select" || keyword === "with" || keyword === "values" || keyword === "table") {
    return { statementKind: "query", comparable: true };
  }
  if (
    keyword === "load" ||
    keyword === "create" ||
    keyword === "insert" ||
    keyword === "update" ||
    keyword === "delete" ||
    keyword === "drop" ||
    keyword === "alter" ||
    keyword === "set" ||
    keyword === "reset" ||
    keyword === "do" ||
    keyword === "analyze" ||
    keyword === "begin" ||
    keyword === "start" ||
    keyword === "commit" ||
    keyword === "end" ||
    keyword === "abort" ||
    keyword === "rollback" ||
    keyword === "savepoint" ||
    keyword === "release" ||
    keyword === "prepare" ||
    keyword.startsWith("\\")
  ) {
    return { statementKind: "setup", comparable: false };
  }
  return { statementKind: "other", comparable: false };
}

function nonComparableDmlReason(sql: string): NonComparableReason | undefined {
  const words = topLevelWords(sql);
  const dmlIndex = words.findIndex((word) => word === "insert" || word === "update" || word === "delete");
  if (dmlIndex !== -1 && words.slice(dmlIndex + 1).includes("returning")) {
    return "non_comparable_dml_returning";
  }
  return dataModifyingCteReason(sql);
}

function dataModifyingCteReason(sql: string): NonComparableReason | undefined {
  const cleaned = stripLeadingComments(sql);
  if (!/^with\b/iu.test(cleaned)) {
    return undefined;
  }
  const words = allWords(sql);
  for (let index = 0; index < words.length; index++) {
    const word = words[index];
    if (word === "delete" || word === "insert" || word === "update") {
      return words.slice(index + 1).includes("returning")
        ? "non_comparable_dml_returning"
        : "non_comparable_data_modifying_cte";
    }
  }
  return undefined;
}

function allWords(sql: string): string[] {
  const words: string[] = [];
  let i = 0;
  while (i < sql.length) {
    const ch = sql[i];
    if (ch === "'") {
      i = readSingleQuoted(sql, i, isEscapeStringQuote(sql, i))[1];
      continue;
    }
    if (ch === '"') {
      i = readDoubleQuoted(sql, i)[1];
      continue;
    }
    if (ch === "$") {
      const dollar = tryReadDollarQuoted(sql, i);
      if (dollar) {
        i = dollar.end;
        continue;
      }
    }
    if (ch === "-" && sql[i + 1] === "-") {
      const end = sql.indexOf("\n", i + 2);
      i = end === -1 ? sql.length : end + 1;
      continue;
    }
    if (ch === "/" && sql[i + 1] === "*") {
      const end = sql.indexOf("*/", i + 2);
      i = end === -1 ? sql.length : end + 2;
      continue;
    }
    if (/[A-Za-z_]/.test(ch ?? "")) {
      const match = /^[A-Za-z_][A-Za-z0-9_]*/u.exec(sql.slice(i));
      if (match?.[0]) {
        words.push(match[0].toLowerCase());
        i += match[0].length;
        continue;
      }
    }
    i++;
  }
  return words;
}

function topLevelWords(sql: string): string[] {
  const words: string[] = [];
  let depth = 0;
  let i = 0;
  while (i < sql.length) {
    const ch = sql[i];
    if (ch === "'") {
      i = readSingleQuoted(sql, i, isEscapeStringQuote(sql, i))[1];
      continue;
    }
    if (ch === '"') {
      i = readDoubleQuoted(sql, i)[1];
      continue;
    }
    if (ch === "$") {
      const dollar = tryReadDollarQuoted(sql, i);
      if (dollar) {
        i = dollar.end;
        continue;
      }
    }
    if (ch === "-" && sql[i + 1] === "-") {
      const end = sql.indexOf("\n", i + 2);
      i = end === -1 ? sql.length : end + 1;
      continue;
    }
    if (ch === "/" && sql[i + 1] === "*") {
      const end = sql.indexOf("*/", i + 2);
      i = end === -1 ? sql.length : end + 2;
      continue;
    }
    if (ch === "(") {
      depth++;
      i++;
      continue;
    }
    if (ch === ")") {
      depth = Math.max(0, depth - 1);
      i++;
      continue;
    }
    if (depth === 0 && /[A-Za-z_]/.test(ch ?? "")) {
      const match = /^[A-Za-z_][A-Za-z0-9_]*/u.exec(sql.slice(i));
      if (match?.[0]) {
        words.push(match[0].toLowerCase());
        i += match[0].length;
        continue;
      }
    }
    i++;
  }
  return words;
}

function firstKeyword(sql: string): string {
  const cleaned = stripLeadingComments(sql);
  if (cleaned.startsWith("\\")) {
    return "\\";
  }
  return /^[A-Za-z_][A-Za-z0-9_]*/u.exec(cleaned)?.[0]?.toLowerCase() ?? "";
}

function stripLeadingComments(sql: string): string {
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
    return sql.slice(i);
  }
  return "";
}

function readSingleQuoted(sql: string, start: number, escapeString = false): [string, number] {
  let i = start + 1;
  while (i < sql.length) {
    if (escapeString && sql[i] === "\\") {
      i += sql[i + 1] === undefined ? 1 : 2;
      continue;
    }
    if (sql[i] === "'" && sql[i + 1] === "'") {
      i += 2;
      continue;
    }
    if (sql[i] === "'") {
      return [sql.slice(start, i + 1), i + 1];
    }
    i++;
  }
  return [sql.slice(start), sql.length];
}

function isEscapeStringQuote(sql: string, start: number): boolean {
  return /[eE]/.test(sql[start - 1] ?? "") && !/[A-Za-z0-9_$]/.test(sql[start - 2] ?? "");
}

function readDoubleQuoted(sql: string, start: number): [string, number] {
  let i = start + 1;
  while (i < sql.length) {
    if (sql[i] === '"' && sql[i + 1] === '"') {
      i += 2;
      continue;
    }
    if (sql[i] === '"') {
      return [sql.slice(start, i + 1), i + 1];
    }
    i++;
  }
  return [sql.slice(start), sql.length];
}

function tryReadDollarQuoted(sql: string, start: number): { text: string; end: number } | undefined {
  const tag = /^\$[A-Za-z_][A-Za-z0-9_]*\$|^\$\$/u.exec(sql.slice(start))?.[0];
  if (!tag) {
    return undefined;
  }
  const close = sql.indexOf(tag, start + tag.length);
  if (close === -1) {
    return { text: sql.slice(start), end: sql.length };
  }
  return { text: sql.slice(start, close + tag.length), end: close + tag.length };
}

function readBlockComment(sql: string, start: number, path: string): { text: string; end: number } {
  let depth = 1;
  let i = start + 2;
  while (i < sql.length && depth > 0) {
    if (sql[i] === "/" && sql[i + 1] === "*") {
      depth++;
      i += 2;
      continue;
    }
    if (sql[i] === "*" && sql[i + 1] === "/") {
      depth--;
      i += 2;
      continue;
    }
    i++;
  }
  if (depth !== 0) {
    throw new RouteConfigError(`${path}: unterminated block comment`);
  }
  return { text: sql.slice(start, i), end: i };
}

function isRouteExpectation(value: string): value is RouteExpectation {
  return (validRoutes as readonly string[]).includes(value);
}

function readYamlObject(path: string): Record<string, unknown> {
  const parsed = parse(readFileSync(path, "utf8")) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new RouteConfigError(`${path} must contain a YAML object`);
  }
  return parsed as Record<string, unknown>;
}

function objectValue(value: unknown, path: string, required: true, missingMessage?: () => string): Record<string, unknown>;
function objectValue(value: unknown, path: string, required: false): Record<string, unknown> | undefined;
function objectValue(
  value: unknown,
  path: string,
  required: boolean,
  missingMessage?: () => string
): Record<string, unknown> | undefined {
  if (value === undefined || value === null) {
    if (required) {
      throw new RouteConfigError(missingMessage?.() ?? `${path} is required`);
    }
    return undefined;
  }
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new RouteConfigError(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function stringValue(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new RouteConfigError(`${path} must be a non-empty string`);
  }
  return value;
}

function routeValue(value: unknown): RouteExpectation {
  if (typeof value !== "string" || !isRouteExpectation(value)) {
    throw new RouteConfigError(`invalid route expectation ${String(value)}`);
  }
  return value;
}

function executionModeValue(value: unknown): RouteExecutionMode {
  if (
    value === "stock" ||
    value === "extension-auto" ||
    value === "force-fallback" ||
    value === "force-lower"
  ) {
    return value;
  }
  throw new RouteConfigError(`invalid execution mode ${String(value)}`);
}

function resolveExcludedFiles(sqlDir: string, compareConfig: Record<string, unknown>): ComparePostgresExcludedFile[] {
  const rawFiles = compareConfig.exclude_files ?? [];
  const rawReasons = compareConfig.exclude_reasons ?? {};
  if (!Array.isArray(rawFiles) || rawFiles.some((item) => typeof item !== "string")) {
    throw new RouteConfigError("compare_postgres.exclude_files must be a list of strings");
  }
  const reasons = objectValue(rawReasons, "compare_postgres.exclude_reasons", true);
  const files = rawFiles as string[];
  const knownFiles = sqlFiles(sqlDir).map((path) => relativePosix(sqlDir, path));
  const excluded: ComparePostgresExcludedFile[] = [];

  for (const glob of files) {
    validateExcludeGlob(glob);
    const reason = reasons[glob];
    if (typeof reason !== "string" || reason.trim().length === 0) {
      throw new RouteConfigError(`compare_postgres.exclude_files entry ${glob} must have one non-empty reason`);
    }
    const matchedFiles = knownFiles.filter((file) => globMatches(glob, file));
    if (matchedFiles.length === 0) {
      throw new RouteConfigError(`compare_postgres.exclude_files entry ${glob} did not match any SQL file`);
    }
    excluded.push({ glob, reason, matchedFiles });
  }

  for (const reasonGlob of Object.keys(reasons)) {
    if (!files.includes(reasonGlob)) {
      throw new RouteConfigError(`compare_postgres.exclude_reasons entry ${reasonGlob} is unused`);
    }
  }

  return excluded;
}

function unknownWorkloadMessage(
  workload: string,
  tests: Record<string, unknown> | undefined,
  workloads: Record<string, unknown>
): string {
  const valid = [...new Set([...Object.keys(tests ?? {}), ...Object.keys(workloads)])].sort();
  return `unknown compare-postgres workload ${workload}. Valid workloads: ${valid.join(", ")}`;
}

function validateExcludeGlob(glob: string): void {
  if (isAbsolute(glob) || glob.split(/[\\/]+/).includes("..")) {
    throw new RouteConfigError(`invalid compare_postgres exclude glob ${glob}`);
  }
}

function globMatches(glob: string, file: string): boolean {
  const escaped = glob.replace(/[.+^${}()|[\]\\]/g, "\\$&").replaceAll("*", "[^/]*");
  return new RegExp(`^${escaped}$`).test(file);
}

function sqlFiles(dir: string): string[] {
  if (!existsSync(dir)) {
    return [];
  }
  return readdirSync(dir)
    .flatMap((entry) => {
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) {
        return sqlFiles(path);
      }
      return path.endsWith(".sql") ? [path] : [];
    })
    .sort();
}

function relativePosix(root: string, path: string): string {
  return relative(root, path).split(sep).join("/");
}

function normalizeAbsolute(path: string): string {
  return join(dirname(path), basename(path));
}
