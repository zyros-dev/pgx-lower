import { readFileSync } from "node:fs";
import { describe, expect, test } from "vitest";
import {
  classifyComparisonMode,
  compareResultBlocks,
  parseCompareDiagnostics,
  parseResultBlocks
} from "../src/compare-postgres-results.js";

describe("compare-postgres SQL clause scanner", () => {
  test("detects ordered and multiset queries from top-level clauses", () => {
    expect(classifyComparisonMode("SELECT * FROM t ORDER BY id")).toBe("ordered");
    expect(classifyComparisonMode("SELECT * FROM t")).toBe("multiset");
  });

  test("ignores nested order by, comments, literals, and functions", () => {
    const sql = [
      "SELECT order_by_text, func('ORDER BY nope'), (SELECT x FROM y ORDER BY x)",
      "FROM t",
      "/* ORDER BY comment */",
      "-- ORDER BY line comment",
      "WHERE note = $$ORDER BY dollar$$"
    ].join("\n");

    expect(classifyComparisonMode(sql)).toBe("multiset");
  });

  test("ignores ORDER BY inside nested PostgreSQL block comments", () => {
    const sql = "SELECT * FROM t /* outer /* inner */ ORDER BY fake */ LIMIT 1";

    expect(classifyComparisonMode(sql)).toBe("non_comparable_nondeterministic_order");
  });

  test("reports limit without top-level order as nondeterministic", () => {
    expect(classifyComparisonMode("SELECT * FROM t LIMIT 1")).toBe("non_comparable_nondeterministic_order");
    expect(classifyComparisonMode("SELECT * FROM t ORDER BY id LIMIT 1")).toBe("ordered");
  });

  test("ignores order by inside PostgreSQL escape strings", () => {
    expect(classifyComparisonMode(String.raw`SELECT E'abc\'; ORDER BY fake; -- still literal';`)).toBe("multiset");
  });
});

describe("compare-postgres result block parsing", () => {
  test("parses stdout marker blocks without reading stderr diagnostics", () => {
    const stdout = [
      "__PGX_COMPARE_START__ id=q1 order=ordered",
      "id,name",
      "1,Alice",
      "__PGX_COMPARE_END__ id=q1"
    ].join("\n");
    const stderr = "NOTICE:  [PGX-LOWER] [ROUTE:NOTICE] fallback unsupported_function: x\n";

    expect(parseResultBlocks(stdout)).toEqual([
      {
        id: "q1",
        order: "ordered",
        csv: "id,name\n1,Alice\n"
      }
    ]);
    expect(parseCompareDiagnostics(stderr).routeNotices).toHaveLength(1);
  });

  test("parses psql-prefixed diagnostics without dropping original lines", () => {
    const stderr = [
      "psql:/workspace/tests/tpch/sql/tpch.sql:12: NOTICE:  [PGX-LOWER] [ROUTE:NOTICE] fallback unsupported_plan: fixture",
      "psql:/workspace/tests/tpch/sql/tpch.sql:13: WARNING: planner warning",
      "psql:/workspace/tests/tpch/sql/tpch.sql:14: ERROR: runtime error"
    ].join("\n");

    const diagnostics = parseCompareDiagnostics(stderr);

    expect(diagnostics.routeNotices[0]).toContain("fallback unsupported_plan");
    expect(diagnostics.routeNotices[0]).toContain("psql:/workspace");
    expect(diagnostics.warnings).toHaveLength(1);
    expect(diagnostics.errors).toHaveLength(1);
  });

  test("keeps notice-like stdout CSV fields intact", () => {
    const stdout = [
      "__PGX_COMPARE_START__ id=q1 order=multiset",
      "message",
      "\"NOTICE:  [PGX-LOWER] [ROUTE:NOTICE] not a diagnostic\"",
      "__PGX_COMPARE_END__ id=q1"
    ].join("\n");

    expect(parseResultBlocks(stdout)[0]?.csv).toContain("[ROUTE:NOTICE] not a diagnostic");
  });

  test("keeps marker-looking CSV rows inside the current result block", () => {
    const stdout = [
      "__PGX_COMPARE_START__ id=q1 order=ordered",
      "message",
      "__PGX_COMPARE_END__ id=q1",
      "still data",
      "__PGX_COMPARE_END__ id=q1"
    ].join("\n");

    expect(parseResultBlocks(stdout)).toEqual([
      {
        id: "q1",
        order: "ordered",
        csv: "message\n__PGX_COMPARE_END__ id=q1\nstill data\n"
      }
    ]);
  });

  test("keeps start-marker-looking CSV rows inside the current result block", () => {
    const stdout = [
      "__PGX_COMPARE_START__ id=q1 order=ordered",
      "message",
      "__PGX_COMPARE_START__ id=looks_like_data order=ordered",
      "still data",
      "__PGX_COMPARE_END__ id=q1"
    ].join("\n");

    expect(parseResultBlocks(stdout)).toEqual([
      {
        id: "q1",
        order: "ordered",
        csv: "message\n__PGX_COMPARE_START__ id=looks_like_data order=ordered\nstill data\n"
      }
    ]);
  });

  test("keeps mixed marker-looking CSV rows inside the current result block", () => {
    const stdout = [
      "__PGX_COMPARE_START__ id=q1 order=ordered",
      "message",
      "__PGX_COMPARE_END__ id=q1",
      "__PGX_COMPARE_START__ id=looks_like_data order=ordered",
      "still data",
      "__PGX_COMPARE_END__ id=q1"
    ].join("\n");

    expect(parseResultBlocks(stdout)).toEqual([
      {
        id: "q1",
        order: "ordered",
        csv: "message\n__PGX_COMPARE_END__ id=q1\n__PGX_COMPARE_START__ id=looks_like_data order=ordered\nstill data\n"
      }
    ]);
  });

  test("keeps marker-looking rows in later blocks from extending earlier blocks", () => {
    const stdout = [
      "__PGX_COMPARE_START__ id=q1 order=ordered",
      "a",
      "1",
      "__PGX_COMPARE_END__ id=q1",
      "__PGX_COMPARE_START__ id=q2 order=ordered",
      "b",
      "__PGX_COMPARE_END__ id=q1",
      "2",
      "__PGX_COMPARE_END__ id=q2"
    ].join("\n");

    expect(parseResultBlocks(stdout)).toEqual([
      {
        id: "q1",
        order: "ordered",
        csv: "a\n1\n"
      },
      {
        id: "q2",
        order: "ordered",
        csv: "b\n__PGX_COMPARE_END__ id=q1\n2\n"
      }
    ]);
  });

  test("fails malformed marker streams", () => {
    expect(() => parseResultBlocks("__PGX_COMPARE_START__ id=q1 order=bad\nx")).toThrow(/malformed order/);
    expect(() =>
      parseResultBlocks("__PGX_COMPARE_START__ id=q1 order=ordered\nx\n__PGX_COMPARE_END__ id=q2")
    ).toThrow(/marker id mismatch/);
  });

  test("fails duplicate complete result blocks instead of swallowing them as CSV", () => {
    const stdout = [
      "__PGX_COMPARE_START__ id=q1 order=ordered",
      "id",
      "1",
      "__PGX_COMPARE_END__ id=q1",
      "__PGX_COMPARE_START__ id=q1 order=ordered",
      "id",
      "1",
      "__PGX_COMPARE_END__ id=q1"
    ].join("\n");

    expect(() => parseResultBlocks(stdout)).toThrow(/duplicate result block id q1/);
  });
});

describe("compare-postgres result comparison", () => {
  test("uses csv-parse for PostgreSQL CSV parsing", () => {
    const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
      dependencies?: Record<string, string>;
    };
    const source = readFileSync(new URL("../src/compare-postgres-results.ts", import.meta.url), "utf8");

    expect(packageJson.dependencies).toHaveProperty("csv-parse");
    expect(source).toContain('from "csv-parse/sync"');
  });

  test("reports ordered row mismatch with bounded first details", () => {
    const result = compareResultBlocks({
      workload: "fixture",
      profile: "debug",
      sourceFile: "queries.sql",
      statementIndex: 0,
      routeExpectation: "lower",
      stock: { id: "q1", order: "ordered", csv: "id,name\n1,Alice\n2,Bob" },
      extension: { id: "q1", order: "ordered", csv: "id,name\n1,Alice\n2,Rob" }
    });

    expect(result.ok).toBe(false);
    expect(result.failure?.kind).toBe("cell_mismatch");
    expect(result.failure?.preview).toContain("row 2 column name");
  });

  test("distinguishes SQL NULL from a literal backslash-N string", () => {
    const result = compareResultBlocks({
      workload: "fixture",
      profile: "debug",
      sourceFile: "queries.sql",
      statementIndex: 0,
      routeExpectation: "lower",
      stock: { id: "q1", order: "ordered", csv: "v\n\\N" },
      extension: { id: "q1", order: "ordered", csv: "v\n\"\\N\"" }
    });

    expect(result.ok).toBe(false);
    expect(result.failure?.kind).toBe("cell_mismatch");
  });

  test("distinguishes CRLF from LF inside quoted CSV values", () => {
    const stock = parseResultBlocks(
      "__PGX_COMPARE_START__ id=q1 order=ordered\ntext\n\"a\r\nb\"\n__PGX_COMPARE_END__ id=q1\n"
    )[0]!;
    const extension = parseResultBlocks(
      "__PGX_COMPARE_START__ id=q1 order=ordered\ntext\n\"a\nb\"\n__PGX_COMPARE_END__ id=q1\n"
    )[0]!;

    const result = compareResultBlocks({
      workload: "fixture",
      profile: "debug",
      sourceFile: "queries.sql",
      statementIndex: 0,
      routeExpectation: "lower",
      stock,
      extension
    });

    expect(result.ok).toBe(false);
    expect(result.failure?.kind).toBe("cell_mismatch");
  });

  test("preserves a final blank CSV row before the end marker", () => {
    const stock = parseResultBlocks(
      "__PGX_COMPARE_START__ id=q1 order=ordered\nv\n\n__PGX_COMPARE_END__ id=q1\n"
    )[0]!;
    const extension = parseResultBlocks(
      "__PGX_COMPARE_START__ id=q1 order=ordered\nv\n__PGX_COMPARE_END__ id=q1\n"
    )[0]!;

    const result = compareResultBlocks({
      workload: "fixture",
      profile: "debug",
      sourceFile: "queries.sql",
      statementIndex: 0,
      routeExpectation: "lower",
      stock,
      extension
    });

    expect(result.ok).toBe(false);
    expect(result.failure?.kind).toBe("row_count_mismatch");
  });

  test("preserves a final trailing empty CSV field before the end marker", () => {
    const stock = parseResultBlocks(
      "__PGX_COMPARE_START__ id=q1 order=ordered\nc1,c2\nv,\n__PGX_COMPARE_END__ id=q1\n"
    )[0]!;
    const extension = parseResultBlocks(
      "__PGX_COMPARE_START__ id=q1 order=ordered\nc1,c2\nv\n__PGX_COMPARE_END__ id=q1\n"
    )[0]!;

    const result = compareResultBlocks({
      workload: "fixture",
      profile: "debug",
      sourceFile: "queries.sql",
      statementIndex: 0,
      routeExpectation: "lower",
      stock,
      extension
    });

    expect(result.ok).toBe(false);
    expect(result.failure?.kind).toBe("cell_mismatch");
  });

  test("bounds long mismatch preview values", () => {
    const longStock = "s".repeat(2000);
    const longExtension = "e".repeat(2000);
    const result = compareResultBlocks({
      workload: "fixture",
      profile: "debug",
      sourceFile: "queries.sql",
      statementIndex: 0,
      routeExpectation: "lower",
      stock: { id: "q1", order: "ordered", csv: `text\n${longStock}` },
      extension: { id: "q1", order: "ordered", csv: `text\n${longExtension}` }
    });

    expect(result.ok).toBe(false);
    expect(result.failure?.preview.length).toBeLessThan(500);
    expect(result.failure?.preview).not.toContain(longStock);
    expect(result.failure?.preview).toContain("truncated");
  });

  test("reports unordered duplicate-count mismatch", () => {
    const result = compareResultBlocks({
      workload: "fixture",
      profile: "debug",
      sourceFile: "queries.sql",
      statementIndex: 0,
      routeExpectation: "lower",
      stock: { id: "q1", order: "multiset", csv: "id\n1\n1\n2" },
      extension: { id: "q1", order: "multiset", csv: "id\n1\n2\n2" }
    });

    expect(result.ok).toBe(false);
    expect(result.failure?.kind).toBe("multiset_mismatch");
  });

  test("parses CSV fields with commas, quotes, newlines, and trailing spaces", () => {
    const result = compareResultBlocks({
      workload: "fixture",
      profile: "debug",
      sourceFile: "queries.sql",
      statementIndex: 0,
      routeExpectation: "lower",
      stock: { id: "q1", order: "ordered", csv: "text\n\"a,b \"\n\"quote \"\" ok\"\n\"line\nbreak\"" },
      extension: { id: "q1", order: "ordered", csv: "text\n\"a,b \"\n\"quote \"\" ok\"\n\"line\nbreak\"" }
    });

    expect(result.ok).toBe(true);
  });

  test("reports malformed CSV as a comparison failure", () => {
    const result = compareResultBlocks({
      workload: "fixture",
      profile: "debug",
      sourceFile: "queries.sql",
      statementIndex: 0,
      routeExpectation: "lower",
      stock: { id: "q1", order: "ordered", csv: "id,name\n1,\"unterminated" },
      extension: { id: "q1", order: "ordered", csv: "id,name\n1,Alice" }
    });

    expect(result.ok).toBe(false);
    expect(result.failure?.kind).toBe("stock_csv_parse_failed");
    expect(result.failure?.preview).toContain("Quote Not Closed");
  });
});
