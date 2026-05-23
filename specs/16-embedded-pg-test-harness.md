# Spec 16 — Embedded PG test harness for runtime FFI

**Status:** draft → ready
**Type:** foundational capability (no perf claim; enables future TDD on runtime)
**Depends on:** none
**Unblocks:** Spec 17 (type-system alignment) and any future work on
`src/pgx-lower/runtime/*.cpp`

## Problem

We cannot unit-test the runtime FFI surface today. Files in
`src/pgx-lower/runtime/` (NumericConversion, DateRuntime, StringRuntime,
tuple_access) call into PostgreSQL backend APIs — `palloc`, `datumCopy`,
`ereport`, `MemoryContextAlloc`, numeric digit-array helpers, datetime
conversion helpers, varlena macros. The existing unit-test harness
(`tests/unit/test_lowerings/`) uses `pg_stubs.cpp` — 96 lines of fake symbols
that satisfy the linker but don't actually *behave* like PG. That works for
MLIR-lowering tests (which never call into the stubs at runtime) but fails for
runtime code (which does).

Consequences:
- No red/green TDD for type-conversion code — the highest-defect-density
  area of the codebase (most "fast + wrong" bugs live here).
- The M-site cleanup work in Spec 17 cannot proceed honestly: a stub that
  approximates `ereport` is *not* the test of whether our code handles real PG
  semantics.
- Agent productivity is bottlenecked: failing-test-first discipline is
  impossible when the test framework cannot link against the code under test.

## Goal

Stand up a CMake test-support library that links **real** PG backend object
files for the narrow subset needed by the runtime FFI, so unit tests can
exercise `numeric_to_i128`, `pg_timestamp_to_unix_micros`, varlena helpers,
etc. against authentic PG behavior.

**Acceptance**: at the end of this spec, the following all hold:

1. New CMake target `pg_test_support` links libpgcommon.a + libpgport.a + a
   curated set of PG backend `.o` files (memory context, numeric, datetime,
   varlena, elog).
2. New `test_runtime_init()` shim initializes `MemoryContext` and the PG
   error stack so `palloc` works and `ereport(ERROR)` longjmps cleanly.
3. New `tests/unit/test_runtime/` directory with a working
   `test_numeric_conversion` binary linked against `pg_test_support`,
   containing at minimum:
   - A round-trip test: `i128 → Numeric → i128` for representative scales.
   - A scale-adjustment test exercising the `*=10` loop in `numeric_to_i128`.
   - A negative test: `ereport(ERROR)` from inside the runtime is caught
     by gtest as a test failure (not a process abort).
4. `just utest` passes (`test_numeric_conversion` joins the 5 existing
   tests; total ≥6 green).
5. The existing 5 tests in `test_lowerings/` still pass (no regression).
6. README in `tests/unit/test_runtime/` documents the link strategy + how
   to add a new runtime test file.

**Out of scope** (deliberate — do *not* expand):
- Refactoring runtime code itself (that's Spec 17).
- Testing `tuple_access.cpp` (23 PG-call sites; pull catalog-cache; saved
  for a follow-up if needed — `test_numeric_conversion` is the proof point).
- Mocking the SPI / catalog cache / transaction manager (out of scope by
  design; integration tests cover those paths).

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ pg_test_support  (new CMake STATIC lib)                  │
│  ├─ libpgcommon.a                                        │
│  ├─ libpgport.a                                          │
│  └─ curated PG backend .o files:                         │
│       utils/mmgr/aset.o    (palloc / MemoryContext)      │
│       utils/mmgr/mcxt.o    (MemoryContextInit etc.)      │
│       utils/adt/numeric.o  (numeric digit array)         │
│       utils/adt/datetime.o (date/timestamp)              │
│       utils/adt/varlena.o  (text/bpchar/varchar)         │
│       utils/error/elog.o   (ereport + PG_TRY/PG_CATCH)   │
│       ~5 more discovered iteratively via linker errors   │
│                                                          │
│ pg_test_init.cpp:                                        │
│   pg_test_init() — call once at gtest fixture SetUp.     │
│     - MemoryContextInit()                                 │
│     - CurrentMemoryContext = TopMemoryContext            │
│     - install PG_exception_stack so ereport longjmps    │
│       into a gtest-catchable handler                     │
└──────────────────────────────────────────────────────────┘
        ↓ linked by
   tests/unit/test_runtime/test_numeric_conversion
   (and future test_date, test_varlena, ...)
```

## Components

### Component 1 — `tests/unit/pg_test_support/CMakeLists.txt`

A new STATIC library target. Discovers PG build artifacts via the existing
`PostgreSQL_*` CMake variables (already set up for the extension build).

Reference paths in container:
- `${PostgreSQL_LIBRARY_DIR}/libpgcommon.a`
- `${PostgreSQL_LIBRARY_DIR}/libpgport.a`
- `${PostgreSQL_SRC_DIR}/src/backend/utils/mmgr/aset.o`
- (and the rest, listed above)

Iteration loop: link, read linker error, add the missing `.o`, repeat.
Expect 5–10 iterations.

### Component 2 — `tests/unit/pg_test_support/pg_test_init.{h,cpp}`

```cpp
// pg_test_init.h
namespace pgx::testing {
    // Call once per test fixture SetUp(). Idempotent.
    void pg_test_init();

    // RAII helper: runs `fn` inside a PG_TRY block; if ereport(ERROR)
    // fires, returns the error message as std::string instead of
    // longjmping out of the test.
    std::optional<std::string> catch_ereport(std::function<void()> fn);
}
```

### Component 3 — `tests/unit/test_runtime/test_numeric_conversion.cpp`

Gtest binary, links `pg_test_support` + the existing runtime source
(`src/pgx-lower/runtime/NumericConversion.cpp`).

Test cases (minimum):
1. `RoundTrip_Zero`: i128(0, scale=0) → Numeric → i128 == 0
2. `RoundTrip_Positive`: i128(123456789, scale=4) → Numeric "12345.6789" → i128
3. `RoundTrip_Negative`: same, sign preserved
4. `ScaleAdjustment_Positive`: numeric "1.5" with target_scale=4 → 15000
5. `ScaleAdjustment_Negative`: numeric "12345" with target_scale=-2 → 123
6. `ErrorOnNaN`: `numeric_to_i128(NaN_datum, 0)` logs warning + returns 0
   (current behavior — pinning it)
7. `LargeValue`: near-i128-bounds round-trip

### Component 4 — `tests/unit/test_runtime/CMakeLists.txt`

Wires `test_numeric_conversion` as a ctest target. Add to top-level
`tests/unit/CMakeLists.txt` as a new `add_subdirectory(test_runtime)`.

### Component 5 — `tests/unit/test_runtime/README.md`

One page: how the link strategy works, how to add a new test file (template),
known issues (catalog-cache-dependent code is out of scope).

## Risks & open questions

| Risk | Mitigation |
|---|---|
| Linker errors pull in arbitrarily many `.o` files (cascade) | Cap iterations; if `.o` count exceeds 20, stop and reassess — likely we're touching catalog-dependent code we shouldn't be |
| PG_TRY/PG_CATCH machinery requires more setup than expected | Fallback: skip `ErrorOnNaN` test, mark deferred — the rest still proves the harness |
| `aset.o` requires `pg_malloc` / signal infra | Already in libpgcommon/libpgport; should be covered |
| Numeric digit-array constants differ between PG versions | Pin to PG 17.6 (our build); document version dependency in README |

## Testing strategy

- **Red phase**: write `test_numeric_conversion.cpp` with all 7 cases.
  Confirm they fail because the harness doesn't exist yet (link errors).
- **Green phase**: build out `pg_test_support`, iterate on link errors,
  watch tests go green one by one.
- **Refactor phase**: clean up the link list, document each `.o` choice in
  a CMake comment.

## Implementation order

Each is its own commit on the feature branch:

1. New worktree off `main` → `spec-16-embedded-pg-test-harness`.
2. Scaffold `tests/unit/pg_test_support/` skeleton with empty lib.
3. Write `test_numeric_conversion.cpp` with all 7 cases → confirm link fail.
4. Iterate: add PG `.o` files until link succeeds. Commit each batch.
5. Write `pg_test_init.cpp` + `catch_ereport` helper.
6. Run tests → fix bugs → green.
7. Update `tests/unit/CMakeLists.txt` + write README.
8. `just utest` → 6/6 green. `just compile` → extension still builds.
9. PR to `main`. Title: `feat(test): embedded PG test harness for runtime FFI (spec 16)`.

## Wiki note

Brief entry added to `pgx-lower-architecture.md` under a new "Testing"
section: "Runtime FFI is unit-testable as of Spec 16 via `pg_test_support`
(real PG `.o` files, not stubs). See `tests/unit/test_runtime/README.md`."
