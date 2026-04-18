# Spec 10 — Row-to-column transpose at scan time

**Tier:** 3 (SIMD layer, paired with 09)
**Stack on:** 09 (or land together)
**Blocks:** —
**Estimated effort:** 1 week

## Goal

Stage a chunk of N tuples (8, 16, or 32) into per-column stack/scratch
buffers so vector ops in spec 09 can read contiguous column data via
`vector.transfer_read`. Without this, the row-major heap layout forces
vectorisation back to gather operations, which throw away most of the win.

## Background

PostgreSQL's heap stores tuples row-major. After `heap_deform_tuple`
(used by spec 05's bulk decode), the values for one tuple sit in a `Datum*`
array in tuple-major order — still not what vector ops want.

DuckDB and Vitesse DB use columnar storage end-to-end so this is free.
We're not changing storage, so we transpose dynamically.

The transpose cost is O(N × cols) per chunk, but cols is fixed at codegen
time and the loop unrolls. The win is that downstream vector ops execute
at full SIMD throughput.

## Design

### 1. Strip-mined scan loop structure

Combine spec 05's bulk decode with spec 09's vector ops:

```mlir
// Outer loop: chunks of CHUNK_SIZE tuples
scf.for %chunk_start = 0 to N step %CHUNK_SIZE {
  // Stack-allocate column buffers for this chunk
  %col0 = memref.alloca : memref<8xi64>
  %col1 = memref.alloca : memref<8xi128>
  // ... per column the query touches

  // Inner decode-and-transpose loop (unrolled)
  scf.for %i = 0 to %CHUNK_SIZE step 1 {
    %tuple = read_next_tuple
    %v0 = decode_col_0 %tuple
    %v1 = decode_col_1 %tuple
    memref.store %v0, %col0[%i]
    memref.store %v1, %col1[%i]
  }

  // Now operate on columns vectorisedly
  %vec0 = vector.transfer_read %col0[0] : memref<8xi64>, vector<8xi64>
  // ... compute, filter, accumulate
}
// Tail: scalar loop for remaining N % CHUNK_SIZE tuples
```

### 2. Chunk size

Start at **CHUNK_SIZE = 8** (matches a single AVX-512 register for i64,
two registers for i128 decimals). Let it be a compile-time constant
controlled by a single `constexpr` so it's easy to A/B test 8 vs 16 vs 32.

Trade-offs:
- Small chunks (4–8): low transpose cost, low vector throughput.
- Medium (16–32): sweet spot for most queries.
- Large (64+): scratch buffer pressure, tail loop dominates short scans.

### 3. Where the transpose lives

This is a new MLIR pattern, peer to spec 05's bulk decoder. Two ways to
structure:

**(a) Compose with spec 05.** Bulk decode populates the per-tuple struct;
transpose populates the per-chunk columnar buffer. Two passes.

**(b) Fuse.** Single pattern emits the chunked loop with both decode and
transpose inline, skipping the intermediate per-tuple struct.

Prefer (b) for performance; fall back to (a) if the pattern is too complex
to land in one spec. (a) is correct but writes the same data twice.

### 4. Memory allocation

Use `memref.alloca` (stack allocation) for the column buffers — cheap, no
malloc, lifetime is the chunk loop body. PG's stack budget is large enough
for chunk_size × max_columns × max_type_size = 32 × 32 × 16B = 16KB, which
is fine.

### 5. Type-specific buffers

Strings remain row-major. The transpose pattern only fires for the
vectorisable column subset (ints, floats, decimals — see spec 09's type
guards). String/varlena columns continue to use the per-tuple decode path
even within the chunk loop. The chunked structure still benefits because
the *vectorisable* column ops run at SIMD speed.

### 6. Index scan paths

Index scans (`mlir::dsa::Lookup` op) typically return sparse, non-sequential
tuples. The transpose makes no sense there. Guard the pattern on
sequential-scan parents only. Index scans get spec 05's bulk decode but no
transpose.

## Files to touch

| File | Change |
|------|--------|
| `src/pgx-lower/lingodb_extensions/ChunkScanPattern.cpp` (new) | Strip-mine + transpose pattern |
| `src/pgx-lower/execution/mlir_setup/mlir_runner_phases.cpp` | Add chunk-scan pass before vectorize-scan pass |
| `src/pgx-lower/runtime/tuple_access.cpp` | Possibly add a chunked `decode_n_tuples` helper if option (a) chosen |

## Don't change

- Storage format.
- Index-scan code path (separate spec if needed).
- The vector ops themselves (spec 09 owns those).

## Acceptance criteria

- Build clean.
- All existing tests pass.
- IR inspection on q06: outer loop has chunk size CHUNK_SIZE, scalar tail
  loop after, columnar buffers allocated via `memref.alloca`.
- A/B (warm, SF=0.01, 4-query subset, **vs. spec-09 branch alone**):
  - **q06: ≥1.3× faster** vs spec 09 without transpose (vector.gather is
    much slower than vector.transfer_read on contiguous memory).
  - q01: ≥1.2× faster vs spec 09 without transpose.
  - q03, q12: any direction; report.
- Chunk-size sweep — required deliverable: A/B at CHUNK_SIZE ∈ {4, 8, 16, 32}.
  Pick the value with best aggregate q01+q06 wall time. Default to it.

## Risks

- Cache pressure on the columnar scratch buffers. 8 columns × 8 lanes ×
  i128 = 1KB, well within L1. 32 columns × 32 lanes × i128 = 16KB, edge of
  L1. Test at the chunk-size sweep.
- Decimal i128 transpose is expensive (16B per element, 2× the bandwidth of
  i64). For decimal-heavy queries (q01), the transpose may underperform
  per-tuple scalar. Capture both numbers; if transpose loses on decimals,
  guard the pattern on column type.
- Tail loop must be correct for any scan length. Test with a table whose
  row count is a prime not divisible by any chunk size (e.g., 1009).

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `10-transpose`.

Required runs:
1. Standard 4-query sweep vs spec 09's branch.
2. Cumulative vs `main`: q01 + q06 + q03 + q12.
3. Chunk-size sweep at CHUNK_SIZE ∈ {4, 8, 16, 32}.
4. Full 22-query validation.

## Rollback

The chunk-scan pass is one entry in the pipeline. Removing it falls back to
spec 09's per-tuple-then-vectorise path (slower but functional).
