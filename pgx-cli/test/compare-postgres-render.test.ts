import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { readComparePostgresManifests, resolveComparePostgresTarget } from "../src/sql-manifest.js";
import { renderComparePostgresScripts } from "../src/compare-postgres-render.js";

function fixtureRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "pgx-compare-render-"));
  mkdirSync(join(root, "tests", "tpch", "sql"), { recursive: true });
  writeFileSync(join(root, "tests", "tpch", "sql", "init_tpch.sql"), "CREATE TABLE t(id int, name text);\n");
  writeFileSync(
    join(root, "tests", "tpch", "sql", "tpch.sql"),
    [
      "LOAD 'pgx_lower';",
      "SET pgx_lower.execution_mode = 'auto';",
      "CREATE TEMP TABLE local_t AS SELECT * FROM t;",
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
      "SELECT * FROM local_t ORDER BY id;",
      "DROP TABLE missing_after_last_query;"
    ].join("\n")
  );
  writeFileSync(
    join(root, "tests", "tpch", "sql", "tpch_no_lower.sql"),
    [
      "/* <<pgx-lower-config>>: auto_should_route_to=ignore id=q_unordered */",
      "SELECT name FROM t;"
    ].join("\n")
  );
  writeFileSync(
    join(root, "tests", "workloads.yaml"),
    `
workloads:
  tpch:
    sql: tests/tpch/sql
    setup: tests/tpch/sql/init_tpch.sql
    compare_postgres:
      expected_comparable_count: 2
tests:
  fixture-correctness:
    workload: tpch
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
  );
  return root;
}

describe("compare-postgres script rendering", () => {
  test("renders deterministic stock and extension scripts for comparable sources", () => {
    const root = fixtureRoot();
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });
    const rendered = renderComparePostgresScripts({
      target,
      manifests: readComparePostgresManifests(target)
    });

    expect(rendered.setupFile).toBe(join(root, "tests", "tpch", "sql", "init_tpch.sql"));
    expect(rendered.files.map((file) => file.sourceFile)).toEqual([
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      join(root, "tests", "tpch", "sql", "tpch_no_lower.sql")
    ]);
    expect(rendered.comparableCount).toBe(2);
    expect(rendered.files[0]?.stockSql).not.toContain("pgx_lower");
    expect(rendered.files[0]?.extensionSql).toContain("LOAD 'pgx_lower'");
    expect(rendered.files[0]?.extensionSql).toContain("SET pgx_lower.execution_mode = 'auto'");
    expect(rendered.files[0]?.stockSql).toContain("SET TimeZone = 'UTC';");
    expect(rendered.files[0]?.stockSql).toContain("__PGX_COMPARE_START__ id=q_ordered order=ordered");
    expect(rendered.files[1]?.stockSql).toContain("__PGX_COMPARE_START__ id=q_unordered order=multiset");
    expect(rendered.files[0]?.stockSql).not.toContain("DROP TABLE missing_after_last_query");
  });

  test("strips commented pgx-lower setup from the stock script", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        "-- local extension setup",
        "LOAD 'pgx_lower.so' /* trailing comment */;",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    const rendered = renderComparePostgresScripts({
      target,
      manifests: readComparePostgresManifests(target)
    });

    expect(rendered.files[0]?.stockSql).not.toContain("pgx_lower");
    expect(rendered.files[0]?.extensionSql).toContain("LOAD 'pgx_lower'");
  });

  test("rejects hidden pgx-lower references in inline setup", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        "ALTER DATABASE postgres SET pgx_lower.execution_mode = 'force_lower';",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    expect(() =>
      renderComparePostgresScripts({
        target,
        manifests: readComparePostgresManifests(target)
      })
    ).toThrow(/setup SQL cannot reference pgx_lower/);
  });

  test("rejects DO setup blocks before comparable queries", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        "DO $$",
        "BEGIN",
        "  EXECUTE 'LOAD ' || quote_literal(chr(112)||chr(103)||chr(120)||chr(95)||chr(108)||chr(111)||chr(119)||chr(101)||chr(114));",
        "END $$;",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    expect(() =>
      renderComparePostgresScripts({
        target,
        manifests: readComparePostgresManifests(target)
      })
    ).toThrow(/setup SQL cannot use DO blocks/);
  });

  test("rejects obfuscated LOAD setup before comparable queries", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        String.raw`LOAD E'pgx\137lower.so';`,
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    expect(() =>
      renderComparePostgresScripts({
        target,
        manifests: readComparePostgresManifests(target)
      })
    ).toThrow(/setup SQL cannot use LOAD/);
  });

  test("rejects escaped pgx-lower references in inline setup", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        String.raw`CREATE FUNCTION f() RETURNS int AS E'$libdir/pgx\137lower', 'ts_test_numeric_add_basic' LANGUAGE C;`,
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    expect(() =>
      renderComparePostgresScripts({
        target,
        manifests: readComparePostgresManifests(target)
      })
    ).toThrow(/setup SQL cannot reference pgx_lower/);
  });

  test("does not preserve pgx-lower SET TO setup in extension scripts", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        "SET pgx_lower.execution_mode TO 'force_fallback';",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    const rendered = renderComparePostgresScripts({
      target,
      manifests: readComparePostgresManifests(target)
    });

    expect(rendered.files[0]?.stockSql).not.toContain("force_fallback");
    expect(rendered.files[0]?.extensionSql).not.toContain("force_fallback");
    expect(rendered.files[0]?.extensionSql).toContain("SET pgx_lower.execution_mode = 'auto';");
  });

  test("rejects comparable queries that configure pgx-lower", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        "/* <<pgx-lower-config>>: auto_should_route_to=ignore id=q_poison */",
        "SELECT set_config('pgx_lower.execution_mode', 'force_fallback', false);",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    expect(() =>
      renderComparePostgresScripts({
        target,
        manifests: readComparePostgresManifests(target)
      })
    ).toThrow(/comparable SQL cannot reference pgx_lower/);
  });

  test("reapplies deterministic GUCs after inline setup before COPY blocks", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        "SET extra_float_digits = 0;",
        "SET client_min_messages = error;",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    const rendered = renderComparePostgresScripts({
      target,
      manifests: readComparePostgresManifests(target)
    });
    const stockSql = rendered.files[0]!.stockSql;
    const setupIndex = stockSql.indexOf("SET extra_float_digits = 0;");
    const copyIndex = stockSql.indexOf("\\echo __PGX_COMPARE_START__ id=q_ordered");
    const reappliedFloatIndex = stockSql.indexOf("SET extra_float_digits = 3;", setupIndex + 1);
    const reappliedMessagesIndex = stockSql.indexOf("SET client_min_messages = notice;", setupIndex + 1);

    expect(reappliedFloatIndex).toBeGreaterThan(setupIndex);
    expect(reappliedFloatIndex).toBeLessThan(copyIndex);
    expect(reappliedMessagesIndex).toBeGreaterThan(setupIndex);
    expect(reappliedMessagesIndex).toBeLessThan(copyIndex);
  });

  test("rejects COPY statements before comparable queries", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_copy */",
        "COPY (SELECT 1) TO STDOUT WITH CSV;",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    expect(() =>
      renderComparePostgresScripts({
        target,
        manifests: readComparePostgresManifests(target)
      })
    ).toThrow(/non-comparable statement before the last comparable query/);
  });

  test("rejects stock mode before rendering extension scripts", () => {
    const root = fixtureRoot();
    const target = {
      ...resolveComparePostgresTarget({ root, workload: "fixture-correctness" }),
      executionMode: "stock" as const
    };

    expect(() =>
      renderComparePostgresScripts({
        target,
        manifests: readComparePostgresManifests(target)
      })
    ).toThrow(/compare-postgres renderer cannot use stock execution mode/);
  });

  test("records nondeterministic queries before expected count validation", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_limit */",
        "SELECT * FROM t LIMIT 1;"
      ].join("\n")
    );
    writeFileSync(join(root, "tests", "tpch", "sql", "tpch_no_lower.sql"), "CREATE TEMP TABLE local_t AS SELECT * FROM t;\n");
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tpch:
    sql: tests/tpch/sql
    setup: tests/tpch/sql/init_tpch.sql
    compare_postgres:
      expected_comparable_count: 1
      exclude_files:
        - tpch_no_lower.sql
      exclude_reasons:
        tpch_no_lower.sql: setup-only fixture for this renderer test
tests:
  fixture-correctness:
    workload: tpch
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });
    const rendered = renderComparePostgresScripts({
      target,
      manifests: readComparePostgresManifests(target)
    });

    expect(rendered.comparableCount).toBe(0);
    expect(rendered.files).toEqual([]);
    expect(rendered.nonComparableQueries).toEqual([
      {
        id: "q_limit",
        sourceFile: join(root, "tests", "tpch", "sql", "tpch.sql"),
        statementIndex: 0,
        reason: "non_comparable_nondeterministic_order"
      }
    ]);
  });

  test("uses relative source stems for duplicate basenames in different directories", () => {
    const root = fixtureRoot();
    mkdirSync(join(root, "tests", "tpch", "sql", "a"), { recursive: true });
    mkdirSync(join(root, "tests", "tpch", "sql", "b"), { recursive: true });
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      "CREATE TEMP TABLE local_t AS SELECT * FROM t;\n"
    );
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch_no_lower.sql"),
      "CREATE TEMP TABLE local_t2 AS SELECT * FROM t;\n"
    );
    writeFileSync(
      join(root, "tests", "tpch", "sql", "a", "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_a */\nSELECT * FROM t ORDER BY id;\n"
    );
    writeFileSync(
      join(root, "tests", "tpch", "sql", "b", "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_b */\nSELECT name FROM t ORDER BY name;\n"
    );
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tpch:
    sql: tests/tpch/sql
    setup: tests/tpch/sql/init_tpch.sql
    compare_postgres:
      expected_comparable_count: 2
      exclude_files:
        - tpch.sql
        - tpch_no_lower.sql
      exclude_reasons:
        tpch.sql: setup-only fixture for this renderer test
        tpch_no_lower.sql: setup-only fixture for this renderer test
tests:
  multi-correctness:
    workload: tpch
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const target = resolveComparePostgresTarget({ root, workload: "multi-correctness" });

    const rendered = renderComparePostgresScripts({
      target,
      manifests: readComparePostgresManifests(target)
    });

    expect(rendered.files.map((file) => file.stem)).toEqual(["a/queries.sql", "b/queries.sql"]);
  });

  test("rejects non-setup non-comparable statements before the last comparable query", () => {
    const root = fixtureRoot();
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      [
        "EXPLAIN SELECT * FROM t;",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_ordered */",
        "SELECT * FROM t ORDER BY id;"
      ].join("\n")
    );
    writeFileSync(
      join(root, "tests", "workloads.yaml"),
      `
workloads:
  tpch:
    sql: tests/tpch/sql
    setup: tests/tpch/sql/init_tpch.sql
tests:
  fixture-correctness:
    workload: tpch
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );
    const target = resolveComparePostgresTarget({ root, workload: "fixture-correctness" });

    expect(() =>
      renderComparePostgresScripts({
        target,
        manifests: readComparePostgresManifests(target)
      })
    ).toThrow(/non-comparable statement before the last comparable query/);
  });
});
