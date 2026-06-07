import { describe, expect, test } from "vitest";
import {
  RouteConfigError,
  assertRoutes,
  normalizeRouteNotices,
  parseSqlManifest,
  validateGlobalIds,
  writeRouteSummary
} from "../src/route-assertions.js";

const routeNotice =
  "NOTICE:  [PGX-LOWER] [ROUTE:NOTICE] fallback unsupported_function: unsupported function generate_series() at Plan.targetlist[0]";

describe("route directive parsing", () => {
  test("attaches a strict directive to the following query statement", () => {
    const manifest = parseSqlManifest({
      path: "tests/tpch/sql/tpch.sql",
      sql: "/* <<pgx-lower-config>>: auto_should_route_to=lower id=tpch_q01 */\nSELECT 1;\n",
      defaultRoute: "fallback",
      requireRouteDirectives: true
    });

    expect(manifest.statements).toEqual([
      {
        path: "tests/tpch/sql/tpch.sql",
        index: 0,
        sql: "SELECT 1;",
        autoShouldRouteTo: "lower",
        id: "tpch_q01"
      }
    ]);
  });

  test("rejects near-miss route comments", () => {
    expect(() =>
      parseSqlManifest({
        path: "bad.sql",
        sql: "/* pgx-lower-config: auto_should_route_to=lower id=x */\nSELECT 1;",
        defaultRoute: "lower",
        requireRouteDirectives: false
      })
    ).toThrow(RouteConfigError);
  });

  test.each([
    "/* <<pgx-lower-config>>: auto_should_route_to=lower unknown=x */\nSELECT 1;",
    "/* <<pgx-lower-config>>: auto_should_route_to=lower auto_should_route_to=fallback */\nSELECT 1;",
    "/* <<pgx-lower-config>>: auto_should_route_to=maybe id=x */\nSELECT 1;",
    "/* <<pgx-lower-config>>: auto_should_route_to lower id=x */\nSELECT 1;"
  ])("rejects malformed directive %s", (sql) => {
    expect(() =>
      parseSqlManifest({
        path: "bad.sql",
        sql,
        defaultRoute: "lower",
        requireRouteDirectives: false
      })
    ).toThrow(RouteConfigError);
  });

  test("requires query directives and ids when configured", () => {
    expect(() =>
      parseSqlManifest({
        path: "missing.sql",
        sql: "SELECT 1;",
        defaultRoute: "lower",
        requireRouteDirectives: true
      })
    ).toThrow(/missing route directive/);

    expect(() =>
      parseSqlManifest({
        path: "missing-id.sql",
        sql: "/* <<pgx-lower-config>>: auto_should_route_to=lower */\nSELECT 1;",
        defaultRoute: "lower",
        requireRouteDirectives: true
      })
    ).toThrow(/requires an id/);
  });

  test("requires directives for comment-prefixed query statements", () => {
    expect(() =>
      parseSqlManifest({
        path: "commented-query.sql",
        sql: "-- query label\nSELECT 1;",
        defaultRoute: "lower",
        requireRouteDirectives: true
      })
    ).toThrow(/missing route directive/);
  });

  test("defaults setup statements to ignore without directives", () => {
    const manifest = parseSqlManifest({
      path: "setup.sql",
      sql: "LOAD 'pgx_lower.so';\nCREATE TABLE t(id int);\nINSERT INTO t VALUES (1);\nSET pgx_lower.execution_mode = 'auto';",
      defaultRoute: "lower",
      requireRouteDirectives: true
    });

    expect(manifest.statements.map((statement) => statement.autoShouldRouteTo)).toEqual([
      "ignore",
      "ignore",
      "ignore",
      "ignore"
    ]);
  });

  test("handles semicolons in quoted strings, identifiers, comments, and dollar strings", () => {
    const manifest = parseSqlManifest({
      path: "quoted.sql",
      sql: [
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q */",
        "SELECT ';' AS semi, $$a;b$$ AS dollar, \"semi;colon\" FROM t -- trailing ;",
        "WHERE name = 'x; y';"
      ].join("\n"),
      defaultRoute: "lower",
      requireRouteDirectives: true
    });

    expect(manifest.statements).toHaveLength(1);
    expect(manifest.statements[0]?.sql).toContain("$$a;b$$");
  });

  test("detects duplicate ids across manifests", () => {
    const first = parseSqlManifest({
      path: "a.sql",
      sql: "/* <<pgx-lower-config>>: auto_should_route_to=lower id=dup */\nSELECT 1;",
      defaultRoute: "lower",
      requireRouteDirectives: true
    });
    const second = parseSqlManifest({
      path: "b.sql",
      sql: "/* <<pgx-lower-config>>: auto_should_route_to=fallback id=dup */\nSELECT 2;",
      defaultRoute: "lower",
      requireRouteDirectives: true
    });

    expect(() => validateGlobalIds([first, second])).toThrow(/duplicate route directive id dup/);
  });
});

describe("route notice assertions", () => {
  test("normalizes only stable route notices", () => {
    const text = [
      "SELECT generate_series(1, 2);",
      routeNotice,
      "NOTICE:  [PGX-LOWER] unrelated debug message",
      "(2 rows)"
    ].join("\n");

    expect(normalizeRouteNotices(text)).toBe(
      ["SELECT generate_series(1, 2);", "NOTICE:  [PGX-LOWER] unrelated debug message", "(2 rows)"].join("\n")
    );
  });

  test("maps notices by statement order and fails lower expectations on fallback", () => {
    const manifest = parseSqlManifest({
      path: "queries.sql",
      sql: [
        "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q1 */",
        "SELECT 1;",
        "/* <<pgx-lower-config>>: auto_should_route_to=fallback id=q2 */",
        "SELECT generate_series(1, 2);"
      ].join("\n"),
      defaultRoute: "lower",
      requireRouteDirectives: true
    });

    const report = assertRoutes({
      runName: "route-test",
      profile: "debug",
      executionMode: "extension-auto",
      manifests: [manifest],
      outputsByPath: new Map([
        [
          "queries.sql",
          ["SELECT 1;", routeNotice, "?column?", "SELECT generate_series(1, 2);", routeNotice].join("\n")
        ]
      ])
    });

    expect(report.observedFallbacks).toBe(2);
    expect(report.failures).toEqual([
      {
        statementId: "q1",
        path: "queries.sql",
        statementIndex: 0,
        reason: "expected lower but observed fallback unsupported_function: unsupported function generate_series()"
      }
    ]);
  });

  test("fails fallback expectations when no fallback notice appears", () => {
    const manifest = parseSqlManifest({
      path: "queries.sql",
      sql: "/* <<pgx-lower-config>>: auto_should_route_to=fallback id=q */\nSELECT 1;",
      defaultRoute: "lower",
      requireRouteDirectives: true
    });

    const report = assertRoutes({
      runName: "route-test",
      profile: "debug",
      executionMode: "extension-auto",
      manifests: [manifest],
      outputsByPath: new Map([["queries.sql", "SELECT 1;\n ?column?\n----------\n        1"]])
    });

    expect(report.failures[0]?.reason).toBe("expected fallback but no fallback notice was observed");
  });

  test("force modes override auto route expectations", () => {
    const manifest = parseSqlManifest({
      path: "queries.sql",
      sql: "/* <<pgx-lower-config>>: auto_should_route_to=lower id=q */\nSELECT 1;",
      defaultRoute: "lower",
      requireRouteDirectives: true
    });

    const fallback = assertRoutes({
      runName: "route-test",
      profile: "debug",
      executionMode: "force-fallback",
      manifests: [manifest],
      outputsByPath: new Map([["queries.sql", "SELECT 1;"]])
    });
    const lower = assertRoutes({
      runName: "route-test",
      profile: "debug",
      executionMode: "force-lower",
      manifests: [manifest],
      outputsByPath: new Map([["queries.sql", ["SELECT 1;", routeNotice].join("\n")]])
    });

    expect(fallback.failures[0]?.reason).toBe("expected fallback but no fallback notice was observed");
    expect(lower.failures[0]?.reason).toContain("expected lower but observed fallback");
  });

  test("force modes override comment-prefixed query statements", () => {
    const manifest = parseSqlManifest({
      path: "queries.sql",
      sql: "/* <<pgx-lower-config>>: auto_should_route_to=not_asserted id=q */\n-- query label\nSELECT 1;",
      defaultRoute: "not_asserted",
      requireRouteDirectives: true
    });

    const report = assertRoutes({
      runName: "route-test",
      profile: "debug",
      executionMode: "force-fallback",
      manifests: [manifest],
      outputsByPath: new Map([["queries.sql", "-- query label\nSELECT 1;"]])
    });

    expect(report.failures[0]?.reason).toBe("expected fallback but no fallback notice was observed");
  });

  test("writes a compact markdown summary", () => {
    const summary = writeRouteSummary({
      runName: "pgx-regression",
      profile: "debug",
      executionMode: "extension-auto",
      routeCounts: { lower: 1, fallback: 1, ignore: 0, not_asserted: 0 },
      observedFallbacks: 1,
      failures: [
        {
          statementId: "q1",
          path: "queries.sql",
          statementIndex: 0,
          reason: "expected lower but observed fallback unsupported_function: unsupported function generate_series()"
        }
      ]
    });

    expect(summary).toContain("# Route Summary: pgx-regression");
    expect(summary).toContain("- Execution mode: extension-auto");
    expect(summary).toContain("q1");
    expect(summary).toContain("queries.sql");
  });
});
