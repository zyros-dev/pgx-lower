# Spec 16 — Unit tests for runtime FFI via pg_regress UDFs

**Status:** draft (replaces earlier "embedded PG test harness" draft)
**Type:** foundational capability (unblocks Spec 17 type-system work)

## Problem

`src/pgx-lower/runtime/*.cpp` is currently untested at unit level. The existing
gtest setup in `tests/unit/test_lowerings/` uses `pg_stubs.cpp` — fake PG symbols
that satisfy the linker but don't behave like PG. Adequate for pure-MLIR code,
inadequate for runtime FFI (numeric, varlena, date, tuple_access).

## Why the previous "embedded PG" attempt was wrong

Earlier draft of this spec bundled `src/backend/*.o` into a `libpgbackend.a`
and tried to link it into a standalone gtest binary. After 15+ iterations and
multiple Docker image rebuilds, the linker fight cascaded indefinitely
(`progname`, `pqsignal`, ICU, `unicode_is_normalized_quickcheck`, duplicate
`fsync_fname` from `fd.o` vs `file_utils.o`, etc.) because PG's `postgres`
binary is linked from a *curated* set of objects, not a `find | ar` blob.

Research into how three established PG-internal projects handle this:

| Project | Approach |
|---|---|
| TimescaleDB | Test C files → OBJECT LIBRARY → linked into extension `.so` (Debug only) → SQL UDFs → pg_regress |
| Citus | Same: test code part of `citus.so`, exposed as UDFs |
| pgrx | Spawn real PG, load extension `.so`, tests run in-process |
| libpg_query | Standalone is possible but requires libclang AST dependency analysis (weeks of work, out of scope) |

**Verdict:** no one bundles `src/backend/*.o` into a gtest binary. The
established pattern is to compile test C code *into the extension itself* and
invoke via pg_regress.

## Unit vs integration distinction

pg_regress is just the *runner*. The tests themselves are unit-scoped:

```
Unit test (this spec):       SELECT test.numeric_to_i128_basic();
                                 └─→ ts_test_numeric_to_i128() {
                                        Datum d = build_numeric(...);
                                        if (numeric_to_i128(d, 4) != 15000)
                                            elog(ERROR, "...");
                                     }
                             ↑ exercises ONE C function, controlled inputs
                               and assertions — unit, runner-agnostic.

Integration test (existing): SELECT * FROM lineitem WHERE l_quantity > 10;
                             ↑ exercises full parser→planner→executor pipeline.
```

PostgreSQL's own developers use pg_regress for what they call "regression
tests" but are effectively unit tests of single SQL features. Same pattern.

**Cost we accept**: tests run slower than a pure gtest binary (need a live PG
per suite, ~seconds of startup amortized), less parallel, failure output is
`elog` text rather than a gtest stack trace.

## Goal

A working unit-test harness for runtime FFI code, modeled directly on
TimescaleDB:

1. New CMake OBJECT LIBRARY `pgx_lower_tests` containing the test C files.
2. Linked into `pgx_lower.so` ONLY in Debug builds (guarded by
   `if(CMAKE_BUILD_TYPE MATCHES Debug)`).
3. Test functions registered as SQL UDFs via `PG_FUNCTION_INFO_V1()`.
4. New pg_regress test file(s) that invoke them.
5. Run via the existing `just test` recipe — no new runner.

**Acceptance**:
- `just test` runs all existing tests + at least one new `unit_numeric` test
  exercising `numeric_to_i128` with a constructed Numeric Datum.
- The new test asserts at minimum: zero, positive integer, negative integer,
  fractional rescale-up, scale-down by division, large value (6 cases — same
  as the gtest cases on the abandoned branch, ported to UDF form).
- Test failures produce `elog(ERROR, ...)` output that pg_regress reports as
  a regression.
- Adding a new runtime test = (a) write a C function in `src/pgx-lower/test/`,
  (b) add it to the OBJECT LIBRARY, (c) add a SQL line + expected output.
  No CMake link-list edits per test.
- Release builds DO NOT include the test code (verify: `nm pgx_lower.so` in
  Release mode shows no `ts_test_*` symbols).

**Out of scope**:
- Refactoring runtime code (Spec 17).
- Catalog-cache-dependent paths (those are integration tests by nature).
- Performance benchmarks for tests (this is correctness only).

## Implementation steps

1. **Revert the failed approach** — `git revert` (or remove) the
   `libpgbackend.a` Dockerfile change, `tests/unit/test_runtime/`, and the
   `add_subdirectory(test_runtime)` line in `tests/unit/CMakeLists.txt`.
2. **Create test source dir** `src/pgx-lower/test/` (new), with
   `numeric_tests.c` containing the 6 PG_FUNCTION_INFO_V1 entry points.
3. **Add OBJECT LIBRARY** in `src/CMakeLists.txt` (or wherever the extension
   is wired):
   ```cmake
   if(CMAKE_BUILD_TYPE MATCHES Debug)
     add_library(pgx_lower_tests OBJECT ${TEST_SOURCES})
     set_target_properties(pgx_lower_tests PROPERTIES POSITION_INDEPENDENT_CODE ON)
     target_link_libraries(pgx_lower PRIVATE $<TARGET_OBJECTS:pgx_lower_tests>)
   endif()
   ```
4. **SQL test file** `tests/regress/sql/unit_numeric.sql`:
   ```sql
   CREATE OR REPLACE FUNCTION test.numeric_to_i128_basic() RETURNS VOID
     AS 'MODULE_PATHNAME', 'ts_test_numeric_to_i128_basic' LANGUAGE C VOLATILE;
   SELECT test.numeric_to_i128_basic();
   -- ... 5 more
   ```
5. **Expected output** `tests/regress/expected/unit_numeric.out`: just shows
   the SELECT succeeded (no `elog(ERROR)` fired).
6. **Wire into pg_regress** — add `unit_numeric` to the test schedule.
7. **Verify**: `just test` includes the new test and passes; Release build
   excludes the test symbols.

## Risk

| Risk | Mitigation |
|---|---|
| Test code accidentally linked into Release | CMake guard + explicit `nm` check in CI |
| Test C function fails silently if assertion macro is wrong | Use `elog(ERROR, ...)` directly; pg_regress aborts on any error output |
| Test schema (`test.*`) conflicts with user-created schema | Use `IF NOT EXISTS` + drop at end of test file |
| Adding tests requires extension rebuild | True; acceptable since extension rebuilds are already in the dev loop |

## Wiki update

`pgx-lower-architecture.md` "Testing" section: rewrite to describe the UDF
pattern (replacing the abandoned embedded-PG approach).
