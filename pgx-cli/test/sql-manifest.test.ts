import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import {
  RouteConfigError,
  parseSqlManifest,
  readComparePostgresManifests,
  resolveComparePostgresTarget
} from "../src/sql-manifest.js";

function tempRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "pgx-sql-manifest-"));
  mkdirSync(join(root, "tests", "sql"), { recursive: true });
  return root;
}

function writeWorkloads(root: string, body: string): void {
  mkdirSync(join(root, "tests"), { recursive: true });
  writeFileSync(join(root, "tests", "workloads.yaml"), body);
}

describe("shared SQL manifest parsing", () => {
  test("classifies comparable queries and setup statements", () => {
    const manifest = parseSqlManifest({
      path: "queries.sql",
      sql: [
        "LOAD 'pgx_lower';",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=s */",
        "SELECT 1;",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=w */",
        "WITH x AS (SELECT 1) SELECT * FROM x;",
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=v */",
        "VALUES (1);",
        "ANALYZE;"
      ].join("\n"),
      defaultRoute: "fallback",
      requireRouteDirectives: true
    });

    expect(manifest.statements.map((statement) => [statement.statementKind, statement.comparable])).toEqual([
      ["setup", false],
      ["query", true],
      ["query", true],
      ["query", true],
      ["setup", false]
    ]);
  });

  test("classifies transaction command forms as setup", () => {
    const manifest = parseSqlManifest({
      path: "transactions.sql",
      sql: [
        "START TRANSACTION;",
        "SAVEPOINT before_query;",
        "RELEASE SAVEPOINT before_query;",
        "PREPARE TRANSACTION 'before-query';",
        "COMMIT PREPARED 'before-query';",
        "ABORT;",
        "END;"
      ].join("\n"),
      defaultRoute: "not_asserted",
      requireRouteDirectives: false
    });

    expect(manifest.statements.map((statement) => statement.statementKind)).toEqual([
      "setup",
      "setup",
      "setup",
      "setup",
      "setup",
      "setup",
      "setup"
    ]);
  });

  test("classifies TABLE statements as comparable queries", () => {
    const manifest = parseSqlManifest({
      path: "queries.sql",
      sql: "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_table */\nTABLE t;",
      defaultRoute: "lower",
      requireRouteDirectives: true
    });

    expect(manifest.statements[0]?.statementKind).toBe("query");
    expect(manifest.statements[0]?.comparable).toBe(true);
  });

  test("records DML returning as non-comparable instead of setup", () => {
    const manifest = parseSqlManifest({
      path: "returning.sql",
      sql: "INSERT INTO t VALUES (1) RETURNING *;",
      defaultRoute: "not_asserted",
      requireRouteDirectives: false
    });

    expect(manifest.statements[0]?.statementKind).toBe("other");
    expect(manifest.statements[0]?.nonComparableReason).toBe("non_comparable_dml_returning");
  });

  test("records WITH DML returning as non-comparable", () => {
    const manifest = parseSqlManifest({
      path: "returning.sql",
      sql: "WITH moved AS (SELECT 1) DELETE FROM t RETURNING *;",
      defaultRoute: "not_asserted",
      requireRouteDirectives: false
    });

    expect(manifest.statements[0]?.statementKind).toBe("other");
    expect(manifest.statements[0]?.nonComparableReason).toBe("non_comparable_dml_returning");
  });

  test("records data-modifying CTE returning as non-comparable", () => {
    const manifest = parseSqlManifest({
      path: "returning.sql",
      sql: "WITH moved AS (DELETE FROM t RETURNING *) SELECT * FROM moved;",
      defaultRoute: "not_asserted",
      requireRouteDirectives: false
    });

    expect(manifest.statements[0]?.statementKind).toBe("other");
    expect(manifest.statements[0]?.nonComparableReason).toBe("non_comparable_dml_returning");
  });

  test("records data-modifying CTE without returning as non-comparable", () => {
    const manifest = parseSqlManifest({
      path: "returning.sql",
      sql: "WITH deleted AS (DELETE FROM t) SELECT 1;",
      defaultRoute: "not_asserted",
      requireRouteDirectives: false
    });

    expect(manifest.statements[0]?.statementKind).toBe("other");
    expect(manifest.statements[0]?.nonComparableReason).toBe("non_comparable_data_modifying_cte");
  });

  test("classifies COPY as other instead of setup", () => {
    const manifest = parseSqlManifest({
      path: "copy.sql",
      sql: "COPY (SELECT 1) TO STDOUT WITH CSV;",
      defaultRoute: "not_asserted",
      requireRouteDirectives: false
    });

    expect(manifest.statements[0]?.statementKind).toBe("other");
    expect(manifest.statements[0]?.comparable).toBe(false);
  });

  test("keeps semicolons and order by inside PostgreSQL escape strings", () => {
    const manifest = parseSqlManifest({
      path: "escape.sql",
      sql: String.raw`/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_escape */
SELECT E'abc\'; ORDER BY fake; -- still literal';`,
      defaultRoute: "lower",
      requireRouteDirectives: true
    });

    expect(manifest.statements).toHaveLength(1);
    expect(manifest.statements[0]?.statementKind).toBe("query");
    expect(manifest.statements[0]?.sql).toContain("ORDER BY fake");
  });

  test("ignores DML returning inside PostgreSQL escape strings", () => {
    const manifest = parseSqlManifest({
      path: "escape.sql",
      sql: String.raw`SELECT E'abc\'; DELETE FROM t RETURNING *; -- still literal';`,
      defaultRoute: "not_asserted",
      requireRouteDirectives: false
    });

    expect(manifest.statements).toHaveLength(1);
    expect(manifest.statements[0]?.statementKind).toBe("query");
    expect(manifest.statements[0]?.nonComparableReason).toBeUndefined();
  });

  test("requires IDs for every comparable compare-postgres statement", () => {
    const root = tempRoot();
    writeFileSync(join(root, "tests", "sql", "queries.sql"), "SELECT 1;");
    writeWorkloads(
      root,
      `
workloads:
  raw:
    sql: tests/sql
tests:
  correctness:
    workload: raw
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: false
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "correctness" });
    expect(() => readComparePostgresManifests(target)).toThrow(/requires an id/);
  });

  test("rejects marker-unsafe directive IDs", () => {
    expect(() =>
      parseSqlManifest({
        path: "queries.sql",
        sql: "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q/1 */\nSELECT 1;",
        defaultRoute: "lower",
        requireRouteDirectives: true
      })
    ).toThrow(/invalid route directive id q\/1/);
  });
});

describe("compare-postgres workload resolution", () => {
  test("resolves tests before raw workloads and keeps route defaults", () => {
    const root = tempRoot();
    writeWorkloads(
      root,
      `
workloads:
  tpch:
    sql: tests/sql
tests:
  tpch-correctness:
    workload: tpch
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "tpch-correctness" });

    expect(target.name).toBe("tpch-correctness");
    expect(target.workloadName).toBe("tpch");
    expect(target.sqlDir).toBe(join(root, "tests", "sql"));
    expect(target.defaultAutoShouldRouteTo).toBe("lower");
    expect(target.requireRouteDirectives).toBe(true);
  });

  test("raw workload resolution uses not_asserted route defaults", () => {
    const root = tempRoot();
    writeWorkloads(
      root,
      `
workloads:
  tpch:
    sql: tests/sql
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "tpch" });

    expect(target.profile).toBe("debug");
    expect(target.executionMode).toBe("extension-auto");
    expect(target.defaultAutoShouldRouteTo).toBe("not_asserted");
  });

  test("raw workload resolution ignores route-check execution defaults", () => {
    const root = tempRoot();
    writeWorkloads(
      root,
      `
workloads:
  tpch:
    sql: tests/sql
    execution_mode: force-lower
    default_auto_should_route_to: lower
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "tpch" });

    expect(target.profile).toBe("debug");
    expect(target.executionMode).toBe("extension-auto");
    expect(target.defaultAutoShouldRouteTo).toBe("not_asserted");
  });

  test.each(["stock", "force-fallback", "force-lower"])(
    "rejects %s execution mode for compare-postgres targets",
    (executionMode) => {
    const root = tempRoot();
    writeWorkloads(
      root,
      `
workloads:
  tpch:
    sql: tests/sql
tests:
  tpch-correctness:
    workload: tpch
    profile: debug
    execution_mode: ${executionMode}
    default_auto_should_route_to: lower
`
    );

    expect(() => resolveComparePostgresTarget({ root, workload: "tpch-correctness" })).toThrow(
      /compare-postgres execution_mode must be extension-auto/
    );
    }
  );

  test("tpch-correctness rejects unpinned SQL files unless they are excluded", () => {
    const root = tempRoot();
    mkdirSync(join(root, "tests", "tpch", "sql"), { recursive: true });
    writeFileSync(join(root, "tests", "tpch", "sql", "init_tpch.sql"), "CREATE TABLE t(id int);\n");
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q1 */\nSELECT * FROM t ORDER BY id;\n"
    );
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch_no_lower.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=ignore id=q2 */\nSELECT * FROM t ORDER BY id;\n"
    );
    writeFileSync(join(root, "tests", "tpch", "sql", "extra.sql"), "CREATE TABLE should_not_be_silent(id int);\n");
    writeWorkloads(
      root,
      `
workloads:
  tpch:
    sql: tests/tpch/sql
    setup: tests/tpch/sql/init_tpch.sql
    compare_postgres:
      expected_comparable_count: 2
tests:
  tpch-correctness:
    workload: tpch
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "tpch-correctness" });
    expect(() => readComparePostgresManifests(target)).toThrow(/unexpected compare-postgres source file extra\.sql/);
  });

  test("tpch-correctness enforces 22 comparable statements per pinned source file", () => {
    const root = tempRoot();
    mkdirSync(join(root, "tests", "tpch", "sql"), { recursive: true });
    writeFileSync(join(root, "tests", "tpch", "sql", "init_tpch.sql"), "CREATE TABLE t(id int);\n");
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q1 */\nSELECT * FROM t ORDER BY id;\n"
    );
    writeFileSync(
      join(root, "tests", "tpch", "sql", "tpch_no_lower.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=ignore id=q2 */\nSELECT * FROM t ORDER BY id;\n"
    );
    writeWorkloads(
      root,
      `
workloads:
  tpch:
    sql: tests/tpch/sql
    setup: tests/tpch/sql/init_tpch.sql
    compare_postgres:
      expected_comparable_count: 2
tests:
  tpch-correctness:
    workload: tpch
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "tpch-correctness" });
    expect(() => readComparePostgresManifests(target)).toThrow(/tpch\.sql expected 22 comparable statements, found 1/);
  });

  test("validates excluded files and reasons", () => {
    const root = tempRoot();
    writeFileSync(join(root, "tests", "sql", "skip.sql"), "SELECT 1;");
    writeWorkloads(
      root,
      `
workloads:
  raw:
    sql: tests/sql
    compare_postgres:
      exclude_files:
        - skip.sql
      exclude_reasons:
        skip.sql: route notice fixture
tests:
  correctness:
    workload: raw
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "correctness" });

    expect(target.excludedFiles).toEqual([{ glob: "skip.sql", reason: "route notice fixture", matchedFiles: ["skip.sql"] }]);
  });

  test("exclusion globs skip every matching SQL file", () => {
    const root = tempRoot();
    writeFileSync(
      join(root, "tests", "sql", "keep.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=keep */\nSELECT 1;"
    );
    writeFileSync(join(root, "tests", "sql", "skip_a.sql"), "SELECT 2;");
    writeWorkloads(
      root,
      `
workloads:
  raw:
    sql: tests/sql
    compare_postgres:
      exclude_files:
        - skip_*.sql
      exclude_reasons:
        skip_*.sql: unsupported fixture
tests:
  correctness:
    workload: raw
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "correctness" });
    const manifests = readComparePostgresManifests(target);

    expect(target.excludedFiles[0]?.matchedFiles).toEqual(["skip_a.sql"]);
    expect(manifests.map((manifest) => manifest.path)).toEqual([join(root, "tests", "sql", "keep.sql")]);
  });

  test("fails non-comparable COPY files that would otherwise be dropped", () => {
    const root = tempRoot();
    writeFileSync(
      join(root, "tests", "sql", "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_keep */\nSELECT 1;"
    );
    writeFileSync(
      join(root, "tests", "sql", "copy.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_copy */\nCOPY (SELECT 1) TO STDOUT WITH CSV;"
    );
    writeWorkloads(
      root,
      `
workloads:
  raw:
    sql: tests/sql
tests:
  correctness:
    workload: raw
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "correctness" });
    expect(() => readComparePostgresManifests(target)).toThrow(/copy\.sql: non-comparable statement 0 must be excluded/);
  });

  test("fails setup-only source files that would otherwise be dropped", () => {
    const root = tempRoot();
    writeFileSync(
      join(root, "tests", "sql", "queries.sql"),
      "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q_keep */\nSELECT 1;"
    );
    writeFileSync(join(root, "tests", "sql", "setup_only.sql"), "CREATE TABLE skipped(id int);");
    writeWorkloads(
      root,
      `
workloads:
  raw:
    sql: tests/sql
tests:
  correctness:
    workload: raw
    profile: debug
    execution_mode: extension-auto
    default_auto_should_route_to: lower
    require_route_directives: true
`
    );

    const target = resolveComparePostgresTarget({ root, workload: "correctness" });
    expect(() => readComparePostgresManifests(target)).toThrow(/setup_only\.sql: source file has no comparable statements/);
  });

  test.each([
    ["missing reason", "exclude_files: [skip.sql]\n      exclude_reasons: {}"],
    ["unused reason", "exclude_files: []\n      exclude_reasons:\n        skip.sql: unused"],
    ["absolute glob", "exclude_files: [/tmp/skip.sql]\n      exclude_reasons:\n        /tmp/skip.sql: bad"],
    ["parent glob", "exclude_files: [../skip.sql]\n      exclude_reasons:\n        ../skip.sql: bad"],
    ["unmatched glob", "exclude_files: [missing.sql]\n      exclude_reasons:\n        missing.sql: absent"]
  ])("rejects invalid exclusion config: %s", (_name, snippet) => {
    const root = tempRoot();
    writeFileSync(join(root, "tests", "sql", "skip.sql"), "SELECT 1;");
    writeWorkloads(
      root,
      `
workloads:
  raw:
    sql: tests/sql
    compare_postgres:
      ${snippet}
`
    );

    expect(() => resolveComparePostgresTarget({ root, workload: "raw" })).toThrow(RouteConfigError);
  });
});
