import { parse as parseCsv } from "csv-parse/sync";

import type { RouteExpectation } from "./sql-manifest.js";

export type ComparisonMode = "ordered" | "multiset";
export type ComparisonClassification = ComparisonMode | "non_comparable_nondeterministic_order";

export type ResultBlock = {
  id: string;
  order: ComparisonMode;
  csv: string;
};

export type ComparePostgresFailure = {
  workload: string;
  profile: string;
  sourceFile: string;
  queryId: string;
  statementIndex: number;
  comparisonMode: ComparisonMode;
  routeExpectation: RouteExpectation;
  stockRowCount: number;
  extensionRowCount: number;
  kind: string;
  preview: string;
};

export type ComparePostgresComparison = {
  ok: boolean;
  queryId: string;
  stockRowCount: number;
  extensionRowCount: number;
  failure?: ComparePostgresFailure;
};

export type CompareDiagnostics = {
  routeNotices: string[];
  pgxNotices: string[];
  pgxDiagnostics: string[];
  warnings: string[];
  errors: string[];
};

type CsvCell = {
  value: string;
  isNull: boolean;
};

type LineSpan = {
  contentStart: number;
  contentEnd: number;
  nextStart: number;
  content: string;
};

const startRe = /^__PGX_COMPARE_START__\s+id=([A-Za-z0-9_.:-]+)\s+order=(ordered|multiset)$/;
const malformedStartRe = /^__PGX_COMPARE_START__\s+id=([^\s]+)\s+order=([^\s]+)$/;
const endRe = /^__PGX_COMPARE_END__\s+id=([A-Za-z0-9_.:-]+)$/;
const routeNoticeRe = /^NOTICE:\s+\[PGX-LOWER\] \[ROUTE:NOTICE\]/;
const allowedPgxNoticeRe = /^NOTICE:\s+\[PGX-LOWER\] \[(?:ROUTE:NOTICE|DEBUG|INFO)\]/;

export function classifyComparisonMode(sql: string): ComparisonClassification {
  const words = topLevelWords(sql);
  const hasOrderBy = words.some((word, index) => word === "order" && words[index + 1] === "by");
  if (hasOrderBy) {
    return "ordered";
  }
  if (words.includes("limit") || words.includes("offset") || words.includes("fetch")) {
    return "non_comparable_nondeterministic_order";
  }
  return "multiset";
}

export function parseResultBlocks(stdout: string): ResultBlock[] {
  return parseBlocksFrom(stdout, lineSpans(stdout), 0, new Set());
}

function parseBlocksFrom(stdout: string, lines: LineSpan[], startIndex: number, seen: Set<string>): ResultBlock[] {
  const blocks: ResultBlock[] = [];

  for (let i = startIndex; i < lines.length; ) {
    const rawLine = lines[i];
    if (!rawLine) {
      break;
    }
    const line = rawLine.content.trimEnd();
    const start = startRe.exec(line);
    if (!start) {
      const malformed = malformedStartRe.exec(line);
      if (malformed) {
        throw new Error(`malformed order value ${malformed[2]}`);
      }
      const end = endRe.exec(line);
      if (end) {
        throw new Error(`end marker without start for ${end[1]}`);
      }
      i++;
      continue;
    }

    const id = start[1]!;
    if (seen.has(id)) {
      throw new Error(`duplicate result block id ${id}`);
    }
    const candidateEnds: number[] = [];
    let mismatchedEnd: string | undefined;
    for (let j = i + 1; j < lines.length; j++) {
      const candidate = lines[j]!.content.trimEnd();
      const malformed = malformedStartRe.exec(candidate);
      if (malformed) {
        continue;
      }
      const end = endRe.exec(candidate);
      if (!end) {
        continue;
      }
      if (end[1] === id) {
        candidateEnds.push(j);
      } else if (candidateEnds.length === 0) {
        mismatchedEnd = end[1];
      }
    }

    if (candidateEnds.length === 0) {
      if (mismatchedEnd) {
        throw new Error(`marker id mismatch: started ${id} ended ${mismatchedEnd}`);
      }
      throw new Error(`missing end marker for ${id}`);
    }

    const branchErrors: Error[] = [];
    for (const endIndex of candidateEnds) {
      const branchSeen = new Set(seen);
      branchSeen.add(id);
      try {
        return [
          ...blocks,
          {
            id,
            order: start[2] as ComparisonMode,
            csv: stdout.slice(rawLine.nextStart, lines[endIndex]!.contentStart)
          },
          ...parseBlocksFrom(stdout, lines, endIndex + 1, branchSeen)
        ];
      } catch (error) {
        const branchError = error instanceof Error ? error : new Error(String(error));
        if (branchError.message.startsWith("duplicate result block id ")) {
          throw branchError;
        }
        branchErrors.push(branchError);
      }
    }
    throw branchErrors.at(-1) ?? new Error(`missing end marker for ${id}`);
  }
  return blocks;
}

function lineSpans(text: string): LineSpan[] {
  const spans: LineSpan[] = [];
  let start = 0;
  while (start < text.length) {
    let end = start;
    while (end < text.length && text[end] !== "\n" && text[end] !== "\r") {
      end++;
    }
    let nextStart = end;
    if (text[end] === "\r" && text[end + 1] === "\n") {
      nextStart = end + 2;
    } else if (text[end] === "\n" || text[end] === "\r") {
      nextStart = end + 1;
    }
    spans.push({
      contentStart: start,
      contentEnd: end,
      nextStart,
      content: text.slice(start, end)
    });
    start = nextStart;
  }
  return spans;
}

export function parseCompareDiagnostics(stderr: string): CompareDiagnostics {
  const diagnostics: CompareDiagnostics = {
    routeNotices: [],
    pgxNotices: [],
    pgxDiagnostics: [],
    warnings: [],
    errors: []
  };

  for (const line of stderr.split(/\r?\n/).filter((item) => item.length > 0)) {
    const diagnostic = stripPsqlLocationPrefix(line);
    if (diagnostic.includes("[PGX-LOWER]")) {
      diagnostics.pgxDiagnostics.push(line);
    }
    if (routeNoticeRe.test(diagnostic)) {
      diagnostics.routeNotices.push(line);
      continue;
    }
    if (allowedPgxNoticeRe.test(diagnostic)) {
      diagnostics.pgxNotices.push(line);
      continue;
    }
    if (/^WARNING:/.test(diagnostic)) {
      diagnostics.warnings.push(line);
      continue;
    }
    if (/^ERROR:/.test(diagnostic)) {
      diagnostics.errors.push(line);
    }
  }
  return diagnostics;
}

function stripPsqlLocationPrefix(line: string): string {
  return line.replace(/^psql:[^:]+:\d+:\s+/, "");
}

export function compareResultBlocks(options: {
  workload: string;
  profile: string;
  sourceFile: string;
  statementIndex: number;
  routeExpectation: RouteExpectation;
  stock: ResultBlock;
  extension: ResultBlock;
}): ComparePostgresComparison {
  if (options.stock.id !== options.extension.id) {
    return failure(options, 0, 0, "result_id_mismatch", `stock ${options.stock.id} extension ${options.extension.id}`);
  }
  if (options.stock.order !== options.extension.order) {
    return failure(options, 0, 0, "comparison_mode_mismatch", `stock ${options.stock.order} extension ${options.extension.order}`);
  }

  let stock: CsvCell[][];
  let extension: CsvCell[][];
  try {
    stock = parseCsvRows(options.stock.csv);
  } catch (error) {
    return failure(options, 0, 0, "stock_csv_parse_failed", errorMessage(error));
  }
  try {
    extension = parseCsvRows(options.extension.csv);
  } catch (error) {
    return failure(options, 0, 0, "extension_csv_parse_failed", errorMessage(error));
  }
  const stockHeader = stock[0] ?? [];
  const extensionHeader = extension[0] ?? [];
  const stockRows = stock.slice(1);
  const extensionRows = extension.slice(1);

  if (!rowsEqual(stockHeader, extensionHeader)) {
    return failure(
      options,
      stockRows.length,
      extensionRows.length,
      "header_mismatch",
      `stock ${boundedJson(rowDisplay(stockHeader))} extension ${boundedJson(rowDisplay(extensionHeader))}`
    );
  }

  if (options.stock.order === "ordered") {
    return compareOrdered(options, stockHeader, stockRows, extensionRows);
  }
  return compareMultiset(options, stockRows, extensionRows);
}

function compareOrdered(
  options: Parameters<typeof compareResultBlocks>[0],
  header: CsvCell[],
  stockRows: CsvCell[][],
  extensionRows: CsvCell[][]
): ComparePostgresComparison {
  if (stockRows.length !== extensionRows.length) {
    return failure(options, stockRows.length, extensionRows.length, "row_count_mismatch", boundedPreview(stockRows, extensionRows));
  }
  for (let row = 0; row < stockRows.length; row++) {
    const stock = stockRows[row] ?? [];
    const extension = extensionRows[row] ?? [];
    for (let column = 0; column < Math.max(stock.length, extension.length); column++) {
      if (!cellsEqual(stock[column], extension[column])) {
        const name = header[column]?.value ?? `column-${column + 1}`;
        return failure(
          options,
          stockRows.length,
          extensionRows.length,
          "cell_mismatch",
          `row ${row + 1} column ${name}: stock ${boundedJson(cellDisplay(stock[column]))} extension ${boundedJson(cellDisplay(extension[column]))}`
        );
      }
    }
  }
  return { ok: true, queryId: options.stock.id, stockRowCount: stockRows.length, extensionRowCount: extensionRows.length };
}

function compareMultiset(
  options: Parameters<typeof compareResultBlocks>[0],
  stockRows: CsvCell[][],
  extensionRows: CsvCell[][]
): ComparePostgresComparison {
  const stockCounts = rowCounts(stockRows);
  const extensionCounts = rowCounts(extensionRows);
  const keys = [...new Set([...stockCounts.keys(), ...extensionCounts.keys()])].sort();
  for (const key of keys) {
    if ((stockCounts.get(key) ?? 0) !== (extensionCounts.get(key) ?? 0)) {
      return failure(
        options,
        stockRows.length,
        extensionRows.length,
        "multiset_mismatch",
        `row ${boundedText(key)}: stock count ${stockCounts.get(key) ?? 0} extension count ${extensionCounts.get(key) ?? 0}`
      );
    }
  }
  return { ok: true, queryId: options.stock.id, stockRowCount: stockRows.length, extensionRowCount: extensionRows.length };
}

function failure(
  options: Parameters<typeof compareResultBlocks>[0],
  stockRowCount: number,
  extensionRowCount: number,
  kind: string,
  preview: string
): ComparePostgresComparison {
  return {
    ok: false,
    queryId: options.stock.id,
    stockRowCount,
    extensionRowCount,
    failure: {
      workload: options.workload,
      profile: options.profile,
      sourceFile: options.sourceFile,
      queryId: options.stock.id,
      statementIndex: options.statementIndex,
      comparisonMode: options.stock.order,
      routeExpectation: options.routeExpectation,
      stockRowCount,
      extensionRowCount,
      kind,
      preview
    }
  };
}

function parseCsvRows(csv: string): CsvCell[][] {
  if (csv.length === 0) {
    return [];
  }
  return parseCsv(csv, {
    cast: (value: string, context: { quoting?: boolean }) => ({
      value,
      isNull: value === "\\N" && context.quoting !== true
    }),
    relaxColumnCount: true,
    skipEmptyLines: false
  }) as unknown as CsvCell[][];
}

function rowCounts(rows: CsvCell[][]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const row of rows) {
    const key = JSON.stringify(row.map(cellKey));
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

function rowsEqual(left: CsvCell[], right: CsvCell[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((cell, index) => cellsEqual(cell, right[index]));
}

function cellsEqual(left: CsvCell | undefined, right: CsvCell | undefined): boolean {
  if (!left || !right) {
    return left === right;
  }
  return left.isNull === right.isNull && left.value === right.value;
}

function cellKey(cell: CsvCell): string {
  return cell.isNull ? "null" : `text:${cell.value}`;
}

function rowDisplay(row: CsvCell[]): string[] {
  return row.map((cell) => cellDisplay(cell));
}

function cellDisplay(cell: CsvCell | undefined): string {
  if (!cell) {
    return "<missing>";
  }
  return cell.isNull ? "<NULL>" : cell.value;
}

function boundedPreview(stockRows: CsvCell[][], extensionRows: CsvCell[][]): string {
  return `stock rows ${stockRows.length} extension rows ${extensionRows.length}`;
}

function boundedJson(value: unknown): string {
  return boundedText(JSON.stringify(value));
}

function boundedText(value: string): string {
  const limit = 180;
  return value.length <= limit ? value : `${value.slice(0, limit)}... [truncated ${value.length - limit} chars]`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function topLevelWords(sql: string): string[] {
  const words: string[] = [];
  let depth = 0;
  let i = 0;
  while (i < sql.length) {
    const ch = sql[i];
    if (ch === "'") {
      i = readSingleQuoted(sql, i, isEscapeStringQuote(sql, i));
      continue;
    }
    if (ch === '"') {
      i = readDoubleQuoted(sql, i);
      continue;
    }
    if (ch === "$") {
      const dollar = tryReadDollarQuoted(sql, i);
      if (dollar !== undefined) {
        i = dollar;
        continue;
      }
    }
    if (ch === "-" && sql[i + 1] === "-") {
      const end = sql.indexOf("\n", i + 2);
      i = end === -1 ? sql.length : end + 1;
      continue;
    }
    if (ch === "/" && sql[i + 1] === "*") {
      i = readBlockComment(sql, i);
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

function readSingleQuoted(sql: string, start: number, escapeString = false): number {
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
      return i + 1;
    }
    i++;
  }
  return sql.length;
}

function isEscapeStringQuote(sql: string, start: number): boolean {
  return /[eE]/.test(sql[start - 1] ?? "") && !/[A-Za-z0-9_$]/.test(sql[start - 2] ?? "");
}

function readDoubleQuoted(sql: string, start: number): number {
  let i = start + 1;
  while (i < sql.length) {
    if (sql[i] === '"' && sql[i + 1] === '"') {
      i += 2;
      continue;
    }
    if (sql[i] === '"') {
      return i + 1;
    }
    i++;
  }
  return sql.length;
}

function tryReadDollarQuoted(sql: string, start: number): number | undefined {
  const tag = /^\$[A-Za-z_][A-Za-z0-9_]*\$|^\$\$/u.exec(sql.slice(start))?.[0];
  if (!tag) {
    return undefined;
  }
  const close = sql.indexOf(tag, start + tag.length);
  return close === -1 ? sql.length : close + tag.length;
}

function readBlockComment(sql: string, start: number): number {
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
  return i;
}
