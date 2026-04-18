---
name: pgx-lower-versions-and-history
description: Version pinning (LLVM 20, MLIR 20, Clang 20, PG 17.6, Ubuntu 24.04, Clang-only build), known compiler bugs, recurring fragile areas, and lessons mined from git history. Use when bumping versions, hitting compile failures that look version-related, debugging LLVM/MLIR API mismatches, when a fragile-area pattern (decimals, nulls, hash joins) looks familiar, or when planning a refactor in a known-debt area.
---

# Versions and history

This is the short summary. **For the deep dive (release-note citations,
specific PR numbers, full commit chronology, action items per spec) read
`docs/deep-study-versions-and-history.md`** — it's referenced throughout.

## What's pinned

`docker/dev/Dockerfile`:
```
LLVM_VERSION=20    →  apt.llvm.org/llvm.sh 20  →  Clang 20 + LLVM 20 + MLIR 20
PG_VERSION=17.6    →  built from source against LLVM 20
Ubuntu 24.04 base, CMake 3.31.6, GCC 14 (alternative, not used for our build)
```

`CMakeLists.txt`:
- C++20 required.
- **Clang only.** GCC 14.2.0's `-O3` infinite-loops in
  `mlir::reconcileUnrealizedCasts`. Hard `FATAL_ERROR` if not Clang
  (lines 19-35). Don't try to override.
- LLVM/MLIR found via `find_package(LLVM REQUIRED)` and
  `MLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`.
- PG path hardcoded to `/usr/local/pgsql` in
  `cmake/PostgreSQLConfig.cmake:1`.

## LLVM 20 — what matters here

Released 2025-03-04 as 20.1.0. For our use:

- **PassBuilder API source-compatible with 18/19.** The pipeline at
  `jit_execution_engine.cpp:265-365` works as-is.
- **All passes we use are unchanged**: SROA, InstCombine, Promote, LICM (with
  MemorySSA), Reassociate, GVN, SimplifyCFG.
- **`PipelineTuningOptions` defaults unchanged** for `LoopUnrolling`,
  `LoopVectorization`, `SLPVectorization` (we have all three off).
- **`mlir::ExecutionEngine` still uses RuntimeDyld**, not JITLink. Migration
  is for 21+. Spec 03's compile cache plugs into
  `ExecutionEngineOptions::cache` (unchanged in 20).
- **`registerPipelineStartEPCallback`** is the right hook for spec 08 (PG
  bitcode inlining).
- **`llvm::sys::getHostCPUName()`** at `jit_execution_engine.cpp:297` is the
  *correct* way to get host CPU. Bug #130509 affects users of
  `JITTargetMachineBuilder::detectHost()` — we're safe.

For SLP / loop vectorization (spec 02): start with SLP, it's the cheaper win
on our typical IR shape. See deep-study `Part 2`.

## Clang 20 build-flag changes that may bite us

- **`-Wdeprecated-literal-operator` on by default.** `operator"" _foo` →
  warning. Under `-Werror` → error.
- **`-Wenum-constexpr-conversion` no longer suppressible.** Was a warning,
  now a hard error in constexpr contexts.
- **TBAA tightening for incompatible pointer types.** May silently change
  behaviour of strict-aliasing-violating code. Workaround: `-fno-pointer-tbaa`.
  PG itself uses `-fno-strict-aliasing`; consider matching.
- **`-fwrapv` no longer implies `-fwrapv-pointer`.** New flag for pointer
  overflow specifically. `ptr + unsigned_offset < ptr` now optimizes to
  `false`; use `(uintptr_t)ptr + offset` for overflow checks.
- `[[clang::lifetimebound]]` on void-returning fns is now an error (was
  silently ignored).

## MLIR 20 — the one thing to know

**The 1:1 and 1:N dialect-conversion drivers were merged in LLVM 20.**

If a `TypeConverter` declares any 1:N rule (one source type → multiple target
types), every `ConversionPattern` touching that converter must implement the
1:N overload. Otherwise:

```
fatal: pattern '<name>' does not support 1:N conversion
```

This is `report_fatal_error`, not `LogicalResult::failure`. **Aborts the host
process** — from PG's perspective, the backend crashes, not just the query.

Old (still works for 1:1):
```cpp
LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands, ...);
```

New 1:N overload (preferred when converter might split values):
```cpp
LogicalResult matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands, ...);
```

For pgx-lower today: our patterns in `RelAlgToDB`, `DBToStd`, `DSAToStd` are
1:1. If we ever pull in a newer upstream pattern that introduces 1:N rules,
every connected pattern in the same converter call needs the new overload.

Other MLIR 20 changes:
- **`addArgumentMaterialization` deprecated** → use `addSourceMaterialization`.
- **`populateVectorToLLVMConversionPatterns` is now strictly conversion patterns.**
  The rewrite patterns it used to bundle moved out. Spec 09 must split into
  greedy-rewrite phase + dialect-conversion phase.
- **Pointer-element type fully removed.** `LLVM::LLVMPointerType::get(ctx)`
  only — no element type arg.
- **mlir-runner consolidation**: `mlir-cpu-runner` and friends merged into
  `mlir-runner`. Doesn't affect us (we use in-process API).

## PG 17.6 — what matters here

Released 2025-08-14. For our use:

- **APIs we touch are unchanged**: `ExecutorRun_hook`, `PlannedStmt`,
  `heap_getattr`, `heap_deform_tuple`, `palloc`, `MemoryContextSwitchTo`,
  `j2date`, `numeric_in/out`, `tuplesort_begin_heap`, `datumCopy`, `PG_TRY`.
- **CVE-2025-8713 fix** — planner-time permission checks may fire earlier
  than before for view-mediated queries. Our hook's PG_TRY/PG_CATCH should
  tolerate this. If we log "executor saw …" inside the hook, the message can
  fire *before* the hook ever runs.
- **CTE-heavy and GROUP BY-reordered plans have different shapes**. If any
  test asserts on plan structure, re-baseline.
- **MERGE on inheritance parents** is now correct (was wrong/crashing). If
  any test golden assumed the old behavior, update.
- **BRIN multi-ops** distance miscalculation fixed. `REINDEX` if any
  benchmark uses BRIN (TPC-H doesn't by default).

## Two LLVMs in one process — the perennial collision risk

PG 17 includes its own LLVM JIT (`llvmjit.so`), lazy-loaded when
`jit_above_cost` is exceeded *and* `jit = on`.

If pgx-lower's MLIR-bundled LLVM (20) and PG's `llvmjit.so` LLVM are
different majors, you get duplicate `cl::opt` registrations / `RegisterPass<>`
collisions / target registrations clashing → abort at second LLVM init.

**Mitigation in our Docker setup**: PG 17.6 is built against the *same* LLVM
20 the MLIR uses. Verify with:
```bash
pg_config --configure | tr ' ' '\n' | grep -i llvm
llvm-config --version
```

**Backup mitigation**: turn off PG's JIT for our sessions:
```c
SetConfigOption("jit", "off", PGC_USERSET, PGC_S_SESSION);
```

## Recurring fragile areas (recognise these patterns)

When you find yourself working in any of these areas, expect bugs. Read the
relevant section of `docs/deep-study-versions-and-history.md` Part 5 for the
historical commit chain.

| Area | Why fragile | Where to look |
|------|-------------|---------------|
| **Decimal/numeric** | i128 overflow, scale mismatches, buffer-size assumptions. 27 days, 14 commits to stabilise. | `NumericConversion.cpp` — isolated by commit 5295091. Don't touch the conversion logic without tests. |
| **Null handling** | Auto-wrapping NullableType broke at aggregation. Required rearchitect at 60% progress. | NullableType use throughout DB dialect; `db.isnull`, `db.as_nullable`, `db.nullable_get_val`. |
| **Hash joins** | Empty hash table, NULL predicates, planner not picking hash join by default. | `HashJoinTranslator`, `Joins.cpp`. Use `impl="hash"` attribute to force. |
| **Aggregation** | Combining mode, AVG split, type widening. | `AggregationOp.cpp`, `plan_translator_agg.cpp:371` (AVG TODO). |
| **Memory contexts** | MLIR vs PG ownership clashes; double-free; pfree on PG-owned data. | Anywhere `palloc`, `pfree`, `MemoryContextSwitchTo` appears. |
| **PG_TRY / PG_CATCH** | C++ exceptions across setjmp = UB. Cleanup ownership. | `mlir_runner.cpp:99-121`, `JITEngine::execute`. |

## Recurring crash root causes (memory & type system)

From git: 8 distinct crash fixes in August 2025 alone, all on memory/type
boundaries between PG and MLIR. The recurring lessons:

1. **Don't enable PassManager debugging features** (timing, statistics,
   crash reproduction) in a PG extension — they bypass PG memory contexts
   and crash. Commit c869bd5 removed them.
2. **Don't use `CodeGenOptLevel::Default`** without thinking — aggressive
   inlining can break PG calling conventions. We currently use `Default` at
   `mlir_runner.cpp:152`; commit 244532f had to fall back to `None` once
   for a related reason.
3. **Type ID collisions** between PG and MLIR need compiler pragma isolation
   (commit a997aa1).
4. **`heap_freetuple` ownership**: in `POSTGRESQL_EXTENSION` mode, PG owns
   the original tuple, not us. Don't `pfree` it. (Commit 4049a9c — classic
   double-free.)

## Top 10 lessons (from the deep-study doc)

1. Establish type-system constraints upfront.
2. Isolate MLIR memory contexts from PG. Non-negotiable.
3. Never commit debug PrintOps in MLIR lowerings — use `PGX_LOG`.
4. Null semantics require redesign, not patching.
5. Hash joins need explicit planner enforcement (`impl="hash"`).
6. PCH pays for itself — `translation_pch.h` is the existing file.
7. Modularity can be premature — simplify before abstracting.
8. LLVM/MLIR upgrades are breaking changes — test passes independently.
9. Never enable PassManager debugging in production.
10. Benchmark infrastructure requires isolation — use YAML profiles, not
    commit-per-experiment.

## When you're about to bump a version

Read `docs/deep-study-versions-and-history.md` Part 7 (TL;DR action items)
first. Specifically:

- **Bumping LLVM**: audit for `LLVMPointerType::get(ctx, elemTy)` (must be
  opaque pointers); check `cl::opt` registrations; rebuild PG against the
  same major.
- **Bumping MLIR**: audit `addArgumentMaterialization`; check 1:N pattern
  readiness in conversion passes; verify TableGen `gen-pass-decls` outputs.
- **Bumping PG**: re-baseline plan-shape goldens; check planner permission
  error timing; re-test MERGE / BRIN paths.
- **Bumping Clang**: scan for newly-error diagnostics
  (`-Wenum-constexpr-conversion`, `-Wdeprecated-literal-operator`); audit
  pointer-arithmetic overflow checks for `-fwrapv-pointer` semantics.

## Related skills

- `pgx-lower-jit-compilation` — the LLVM PassBuilder usage that the LLVM 20
  notes describe.
- `pgx-lower-mlir-dialects` — the dialect/conversion code that MLIR 20's
  1:N change affects.
- `pgx-lower-execution-path` — the `ExecutorRun_hook` integration point that
  PG 17.6 changes affect.
- `pgx-lower-build-and-test` — Docker, CMake, the Clang enforcement and
  build flag rationale.
