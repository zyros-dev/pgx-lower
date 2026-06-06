export const validRoutes = ["lower", "fallback", "ignore", "not_asserted"] as const;
export type RouteExpectation = (typeof validRoutes)[number];

export type RouteExecutionMode = "stock" | "extension-auto" | "force-fallback" | "force-lower";

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
};

export type SqlManifest = {
  path: string;
  statements: StatementManifestEntry[];
};

export type RouteEvent = {
  statementIndex: number;
  kind: string;
  message: string;
  location: string | undefined;
};

export type RouteFailure = {
  statementId: string | undefined;
  path: string;
  statementIndex: number;
  reason: string;
};

export type RouteReport = {
  runName: string;
  profile: string;
  executionMode: RouteExecutionMode;
  routeCounts: Record<RouteExpectation, number>;
  observedFallbacks: number;
  failures: RouteFailure[];
};

export class RouteConfigError extends Error {}

const directiveRe = /^\/\*\s*<<pgx-lower-config>>:\s*(.*?)\s*\*\/$/s;
const keyValueRe = /^([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)$/;
const routeNoticeRe =
  /^NOTICE:\s+\[PGX-LOWER\] \[ROUTE:NOTICE\] fallback (?<kind>[a-z_]+): (?<message>.*?)(?: at (?<location>.*))?$/;

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

    const query = isQueryStatement(sql);
    if (options.requireRouteDirectives && query && !pending) {
      throw new RouteConfigError(`${options.path}: statement ${statementIndex} missing route directive`);
    }

    const route = pending?.autoShouldRouteTo ?? (isSetupStatement(sql) ? "ignore" : options.defaultRoute);
    if (options.requireRouteDirectives && query && !pending?.id) {
      throw new RouteConfigError(`${options.path}: statement ${statementIndex} requires an id`);
    }

    manifestStatements.push({
      path: options.path,
      index: statementIndex,
      sql,
      autoShouldRouteTo: route,
      id: pending?.id
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

export function normalizeRouteNotices(outputText: string): string {
  return outputText
    .split(/\r?\n/)
    .filter((line) => !routeNoticeRe.test(line))
    .join("\n");
}

export function assertRoutes(options: {
  runName: string;
  profile: string;
  executionMode: RouteExecutionMode;
  manifests: readonly SqlManifest[];
  outputsByPath: ReadonlyMap<string, string>;
}): RouteReport {
  const routeCounts = emptyRouteCounts();
  const failures: RouteFailure[] = [];
  let observedFallbacks = 0;

  for (const manifest of options.manifests) {
    const output = options.outputsByPath.get(manifest.path) ?? "";
    const events = extractRouteEvents(manifest, output);
    const eventsByStatement = groupEventsByStatement(events);

    for (const statement of manifest.statements) {
      const effective = effectiveExpectation(statement, options.executionMode);
      routeCounts[effective]++;

      const statementEvents = eventsByStatement.get(statement.index) ?? [];
      if (statementEvents.length > 0) {
        observedFallbacks++;
      }

      if (effective === "ignore" || effective === "not_asserted" || options.executionMode === "stock") {
        continue;
      }

      if (effective === "lower" && statementEvents.length > 0) {
        failures.push({
          statementId: statement.id,
          path: statement.path,
          statementIndex: statement.index,
          reason: `expected lower but observed fallback ${formatEvent(statementEvents[0])}`
        });
      }

      if (effective === "fallback" && statementEvents.length === 0) {
        failures.push({
          statementId: statement.id,
          path: statement.path,
          statementIndex: statement.index,
          reason: "expected fallback but no fallback notice was observed"
        });
      }
    }
  }

  return {
    runName: options.runName,
    profile: options.profile,
    executionMode: options.executionMode,
    routeCounts,
    observedFallbacks,
    failures
  };
}

export function writeRouteSummary(report: RouteReport): string {
  const lines = [
    `# Route Summary: ${report.runName}`,
    "",
    `- Profile: ${report.profile}`,
    `- Execution mode: ${report.executionMode}`,
    `- Expected lower: ${report.routeCounts.lower}`,
    `- Expected fallback: ${report.routeCounts.fallback}`,
    `- Ignored: ${report.routeCounts.ignore}`,
    `- Not asserted: ${report.routeCounts.not_asserted}`,
    `- Observed fallback statements: ${report.observedFallbacks}`,
    `- Failures: ${report.failures.length}`,
    ""
  ];

  if (report.failures.length > 0) {
    lines.push("## Failures", "");
    for (const failure of report.failures) {
      const id = failure.statementId ?? `statement-${failure.statementIndex}`;
      lines.push(`- ${id} (${failure.path}:${failure.statementIndex}): ${failure.reason}`);
    }
    lines.push("");
  }

  return `${lines.join("\n")}`;
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
      const [text, end] = readSingleQuoted(sql, i);
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

function readSingleQuoted(sql: string, start: number): [string, number] {
  let i = start + 1;
  while (i < sql.length) {
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

function isQueryStatement(sql: string): boolean {
  const keyword = firstKeyword(sql);
  return keyword === "select" || keyword === "with";
}

function isSetupStatement(sql: string): boolean {
  const keyword = firstKeyword(sql);
  return keyword === "load" ||
    keyword === "create" ||
    keyword === "insert" ||
    keyword === "update" ||
    keyword === "delete" ||
    keyword === "drop" ||
    keyword === "alter" ||
    keyword === "set" ||
    keyword === "reset" ||
    keyword === "copy" ||
    keyword === "do" ||
    keyword.startsWith("\\");
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

function extractRouteEvents(manifest: SqlManifest, output: string): RouteEvent[] {
  const lines = output.split(/\r?\n/);
  const events: RouteEvent[] = [];
  let cursor = 0;

  for (const statement of manifest.statements) {
    const firstLine = firstSqlLine(statement.sql);
    const start = findLine(lines, firstLine, cursor);
    if (start === -1) {
      continue;
    }
    const nextStatement = manifest.statements[statement.index + 1];
    const nextStart = nextStatement ? findLine(lines, firstSqlLine(nextStatement.sql), start + 1) : lines.length;
    const end = nextStart === -1 ? lines.length : nextStart;
    for (let i = start + 1; i < end; i++) {
      const match = routeNoticeRe.exec(lines[i] ?? "");
      const groups = match?.groups;
      if (!groups?.kind || !groups.message) {
        continue;
      }
      events.push({
        statementIndex: statement.index,
        kind: groups.kind,
        message: groups.message,
        location: groups.location
      });
    }
    cursor = end;
  }

  return events;
}

function firstSqlLine(sql: string): string {
  return sql.split(/\r?\n/).map((line) => line.trim()).find((line) => line.length > 0) ?? sql.trim();
}

function findLine(lines: readonly string[], expected: string, start: number): number {
  for (let i = start; i < lines.length; i++) {
    if ((lines[i] ?? "").trim() === expected) {
      return i;
    }
  }
  return -1;
}

function groupEventsByStatement(events: readonly RouteEvent[]): Map<number, RouteEvent[]> {
  const grouped = new Map<number, RouteEvent[]>();
  for (const event of events) {
    const list = grouped.get(event.statementIndex) ?? [];
    list.push(event);
    grouped.set(event.statementIndex, list);
  }
  return grouped;
}

function effectiveExpectation(statement: StatementManifestEntry, mode: RouteExecutionMode): RouteExpectation {
  if (mode === "stock") {
    return "not_asserted";
  }
  if (statement.autoShouldRouteTo === "ignore") {
    return "ignore";
  }
  if (mode === "force-fallback" && isQueryStatement(statement.sql)) {
    return "fallback";
  }
  if (mode === "force-lower" && isQueryStatement(statement.sql)) {
    return "lower";
  }
  return statement.autoShouldRouteTo;
}

function formatEvent(event: RouteEvent | undefined): string {
  if (!event) {
    return "unknown";
  }
  return `${event.kind}: ${event.message}`;
}

function emptyRouteCounts(): Record<RouteExpectation, number> {
  return { lower: 0, fallback: 0, ignore: 0, not_asserted: 0 };
}
