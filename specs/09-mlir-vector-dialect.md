# Spec 09 — Add MLIR vector dialect lowerings for scan/filter/agg

**Tier:** 3 (SIMD layer)
**Stack on:** 02 (LLVM auto-vec result must inform scope)
**Blocks:** 11 (LingoDB rebase decision)
**Estimated effort:** 3–4 weeks

## Goal

Emit MLIR `vector` dialect operations for the inner loops of scan, filter,
and aggregation, instead of relying on LLVM auto-vectorization. Manual
vectorization gets us into DuckDB-range numbers on TPC-H Q1, Q6, Q12 — the
queries that benefit most from SIMD.

**Read spec 02's PR description before starting.** If LLVM auto-vec already
delivered most of the win on q01/q06, this spec's scope shrinks (or descopes
entirely). If it didn't, this spec is committed.

## Background

The current pipeline produces `scf::ForOp` per-tuple loops (spec from agent
research):
- Scan loop: `dsa::ForOp` lowered to `scf::ForOp` at
  `src/lingodb/mlir/Conversion/DSAToStd/CollectionIterators.cpp:378`
  (ForIteratorIterationImpl) or line 319 (WhileIteratorIterationImpl).
- Filter: `mlir::scf::IfOp` at
  `src/lingodb/mlir/Conversion/RelAlgToDB/Translators/SelectionOp.cpp:61`
  or `dsa::CondSkipOp` at line 56.
- Aggregation: scalar hashtable per tuple at
  `src/lingodb/mlir/Conversion/RelAlgToDB/Translators/AggregationOp.cpp`.

The MLIR `vector` dialect is **not registered** anywhere. No code uses it.

LLVM-level vectorization is disabled (`jit_execution_engine.cpp:306-308`).
Spec 02 flips that — the relative gain of *manual* vectorization on top of
*auto* vectorization is what this spec measures.

## Design

### 1. Register vector dialect

`src/pgx-lower/execution/mlir_setup/mlir_runner_passes.cpp` (around lines
84-91 from spec 01's hoist work, otherwise wherever the dialect-load list
lives):

```cpp
context.loadDialect<mlir::vector::VectorDialect>();
```

Also register the standard vector→LLVM lowering interface in the JIT engine's
`register_dialects` call.

### 2. Vector pattern for sequential scan

New file `src/pgx-lower/lingodb_extensions/VectorizeScanPattern.cpp` (or
inside `src/lingodb/mlir/Conversion/DSAToStd/`).

Pattern: detect a `dsa::ForOp` over a `dsa::ScanSource` whose body uses only
vectorizable column types (int, float, decimal as i128 — *not* varlena
strings). Rewrite it to emit a strip-mined loop:

```mlir
// Before (per-tuple):
scf.for %i = %lo to %hi step 1 {
  %v = read_column %i
  %r = arith.muli %v, %const : i64
  store %r
}

// After (vectorized in chunks of 8):
scf.for %i = %lo to %hi_main step 8 {
  %vec = vector.transfer_read %column[%i] : memref<?xi64>, vector<8xi64>
  %r   = arith.muli %vec, %vconst : vector<8xi64>
  vector.transfer_write %r, %out[%i] : vector<8xi64>, memref<?xi64>
}
// epilogue: scalar loop for tail
```

This requires column data to be in a `memref<?xT>` — see step 3.

Vector width: start with **8 lanes** for i64 (matches AVX-512), let LLVM
fall back to 4×i64 (AVX2) when the target lacks AVX-512. The `vector` dialect
gives portability for free; LLVM picks the right instruction set.

### 3. Columnar staging buffer

`vector.transfer_read` needs a contiguous memory range. PG tuples are row-major.
Spec 10 (row-to-column transpose) provides this — it stages a chunk of
column values into a stack-allocated `memref<8xi64>` (or similar) before
the vector op.

Spec 09 and 10 are mutually dependent. Land 10 first as a pure no-vectorisation
refactor, then 09 adds the vector ops. Or land them together as one PR.

### 4. Filter vectorization

For boolean predicates over vector data:

```mlir
%cmp_vec = arith.cmpi sgt, %v, %thresh : vector<8xi64>
// %cmp_vec : vector<8xi1>
// Use vector.compress / vector.gather to apply
```

Filter rewrites the `scf::IfOp` body into masked vector operations. For
high-selectivity predicates (most rows pass), this is a clear win. For
low-selectivity (most rows filtered), masked execution loses to a scalar
predicated loop. **Make the pattern selectivity-aware** — use the planner's
row estimate from `Plan->plan_rows / scan->plan_rows` ratio, take the
vectorised path only above ~30% selectivity. Document the threshold.

### 5. Aggregation vectorization

Two cases:
- **Ungrouped aggregate** (e.g., `SELECT sum(x) FROM t`): trivially vectorisable.
  Compute partial sums in a `vector<8xi64>` accumulator, horizontal-reduce
  at end. Big win on q06.
- **GROUP BY aggregate** (e.g., q01): vectorising the hashtable is genuinely
  hard. Do the partial-sum optimisation only when the inner loop *between*
  hashtable insertions is vectorisable (e.g., computing the aggregate
  expression from multiple columns before inserting). Don't try to vectorise
  the hashtable itself in v1.

### 6. Type guards

Guard the vectorisation pattern on:
- Column types: only int{16,32,64}, float, double, decimal-as-i128 (i128
  vectors lower to scalar but still legal).
- No varlena access in the loop body.
- No FFI calls in the loop body (FFI breaks vectorisation entirely).

These guards must check the *post-spec-05-bulk-decode* form of the loop —
the FFI calls are gone, replaced by struct field reads. Check for those.

### 7. Vector→LLVM lowering

MLIR provides `mlir::populateVectorToLLVMConversionPatterns` — wire it into
`createStandardToLLVMPipeline` (`src/lingodb/mlir/Conversion/StandardToLLVM/StandardToLLVM.cpp:79-87`).
Also add `mlir::vector::populateVectorMaskOpLoweringPatterns` and
`populateVectorTransferLoweringPatterns` for transfer_read/write.

## Files to touch

| File | Change |
|------|--------|
| `src/pgx-lower/execution/mlir_setup/mlir_runner_passes.cpp` | Register vector dialect |
| `src/pgx-lower/lingodb_extensions/VectorizeScanPattern.cpp` (new) | Strip-mining pattern |
| `src/pgx-lower/lingodb_extensions/VectorizeFilterPattern.cpp` (new) | Masked filter |
| `src/pgx-lower/lingodb_extensions/VectorizeAggregatePattern.cpp` (new) | Partial-sum reduction |
| `src/pgx-lower/execution/mlir_setup/mlir_runner_phases.cpp` | Add vectorize pass between phase 3a and 3b (or inside 3b) |
| `src/lingodb/mlir/Conversion/StandardToLLVM/StandardToLLVM.cpp:79-87` | Add vector→LLVM patterns |
| `src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp` | Register vector translation interface |

## Don't change

- LLVM auto-vec settings (spec 02 owns those).
- Hash table internals.
- The DB dialect type system.

## Acceptance criteria

- Build clean.
- All existing tests pass.
- IR inspection: q06 should show `vector.transfer_read`, `arith.muli` on
  vector type, and a horizontal reduction at the end. q01's GROUP BY
  expression should show vector arith on the per-group computation.
- A/B (warm, SF=0.01, 4-query subset, **vs. spec-02-baseline**):
  - **q06: ≥40% faster** vs spec 02 alone (the canonical SIMD target).
  - **q01: ≥20% faster** vs spec 02 alone (vectorisable per-group expressions).
  - q03, q12: any direction; report.
- A/B vs unmodified `main` (cumulative win): **q06 ≥2× faster**,
  **q01 ≥30% faster**.
- Branch-prediction profile: IPC should rise (more work per cycle); LLC-load
  rate should fall (vectors fit in L1 better than per-tuple scalar loads).

## Risks

- Vectorising in MLIR before LLVM doubles the chances of hitting LLVM's
  vector-codegen edge cases. Run extensive correctness tests on full 22-query
  TPC-H, not just the 4-query subset.
- The selectivity threshold for filter vectorisation is workload-dependent.
  If you can't pick a safe default, gate filter vectorisation behind a GUC
  initially.
- Decimal vectorisation (i128) lowers to LLVM scalar i128 (no native SIMD
  for 128-bit ints on x86 until AVX-512 adds limited support). The wins on
  decimal columns will be smaller than on i64. Don't promise q01 (mostly
  decimal) the same gain as q06 (mostly i64) — set expectations.

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `09-vec-dialect`.

Required runs:
1. Standard 4-query sweep, comparing this branch vs spec 02's branch
   (relative win) and vs `main` (cumulative win).
2. Branch-prediction profile.
3. Full 22-query validation — bit-identical results.
4. IR inspection — paste vector ops snippet from q06 to PR.

## Rollback

The vectorize patterns are added passes — remove them from the pipeline,
the rest of the code still works. Branch revert is clean.
