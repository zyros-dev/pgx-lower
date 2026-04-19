# Unit tests

gtest-based unit tests for pgx-lower. Build + run via `just utest`.

## What belongs here vs `tests/sql/`

| Kind of change | Test home | Why |
|---|---|---|
| MLIR pass / lowering pattern / dialect op | `tests/unit/test_lowerings/test_*.cpp` | Fast iteration, scoped to the thing you changed; no PG backend needed |
| Pure-computation utilities (type mapping, cost formulas, plan-shape hashing, etc.) | `tests/unit/test_lowerings/test_*.cpp` | Same pattern as above |
| JIT engine lifecycle / pass ordering | `tests/unit/test_lowerings/test_*.cpp` | `jit_execution_engine.cpp` is in the test target |
| **AST translation** (PG plan / expression node → MLIR) | `tests/unit/test_lowerings/test_ast_*.cpp` | Translators take POD PG nodes; tests fabricate a `Const`/`Var`/`Plan` on the stack and pass it through |
| New SQL feature producing a new output shape | `tests/sql/<NN>_*.sql` + `tests/expected/<NN>_*.out` | pg_regress is the output-equivalence gate |
| Runtime FFI (`DateRuntime`, `NumericConversion`, `StringRuntime`, `tuple_access`, `PostgreSQLRuntime`) | pg_regress, not unit tests | These files ARE the PG integration boundary — their value is measured by integration behavior, not in isolation |
| Catalog lookups, memory-context management, anything that reads from real PG state | pg_regress | No meaningful isolation; mocks would be a parallel reimplementation of PG |

## Existing smoke tests (proof the infra works for each area)

- **`test_pipeline_phases.cpp`** — full MLIR pipeline from RelAlg through Standard/LLVM dialects. Covers specs that change lowering passes or pipeline phases (01, 02, 05, 08, 09, 10, 12).
- **`test_boolean_lowering.cpp`** — individual DB-dialect lowering patterns (NOT/AND/OR/complex). Template for pattern-level tests.
- **`test_type_mapping.cpp`** — `lingodb::utility::mlir_type_to_pg_oid`. Template for pure-computation tests that touch PG types; proves `#define` constants (OIDs) from PG headers are usable in unit tests even though the PG runtime isn't linked.
- **`test_ast_const_translation.cpp`** — `postgresql_ast::translate_const`. Template for AST-translation tests: fabricates a PG `Const` POD on the stack, runs it through the translator, asserts the produced MLIR op kind + value. Covers simpler `translate_const` branches (int/bool/float/null/date); the NUMERICOID and TIMESTAMPOID branches are already `#ifdef POSTGRESQL_EXTENSION`-guarded so they skip in the unit-test build.

Add a new file to `test_lowerings/` following one of these patterns. Wire it into `test_lowerings/CMakeLists.txt` (mirror an existing `add_executable` + `add_test` block).

## When you hit a PG-symbol link error

`pg_stubs.cpp` covers the common leaks (catalog helpers, `palloc`, `pstrdup`, `CurrentMemoryContext`, `pg_fprintf`). If a new source file pulls in a symbol that isn't stubbed, two options:

1. **Extend `pg_stubs.cpp`.** Leaf calls that your test doesn't reach → `pg_stub_abort()`. Leaf calls your test does reach incidentally (logging, malloc-ish things) → a minimal real-ish stub (see `MemoryContextAlloc` / `pg_fprintf`).
2. **Guard the callsite.** If the code unconditionally calls a PG function that truly can't run without a backend, wrap that call in `#ifdef POSTGRESQL_EXTENSION` and fall through to a no-op or trivial default in the unit-test branch. See `jit_execution_engine.cpp`'s `PG_TRY`/`PG_CATCH` block for the pattern.

Prefer option 1 for narrow leaks; option 2 when the whole code path is PG-runtime-bound.

## Tests that can't be unit tests

Runtime FFI functions (`runtime::*`, `tuple_access::*`, catalog-reading helpers) interact with real PG datums, tuples, and the memory system. A unit test for them would be a parallel reimplementation of PG — much more code than the thing being tested, and the test asserts nothing real about production behavior. These stay as pg_regress tests; `tests/sql/` is the correct harness.
