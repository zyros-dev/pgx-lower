import {
  RouteConfigError,
  isQueryStatement,
  parseSqlManifest,
  validateGlobalIds,
  validRoutes
} from "./sql-manifest.js";
import type {
  RouteExecutionMode,
  RouteExpectation,
  SqlManifest,
  StatementManifestEntry
} from "./sql-manifest.js";

export {
  RouteConfigError,
  parseSqlManifest,
  validateGlobalIds,
  validRoutes
};
export type {
  RouteExecutionMode,
  RouteExpectation,
  SqlManifest,
  StatementManifestEntry
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

const routeNoticeRe =
  /^NOTICE:\s+\[PGX-LOWER\] \[ROUTE:NOTICE\] fallback (?<kind>[a-z_]+): (?<message>.*?)(?: at (?<location>.*))?$/;

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
