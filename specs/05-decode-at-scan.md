# Spec 05 — Decode at scan, encode at output

**Tier:** 2 (type-gap track foundation)
**Stack on:** `main` (independent of compile-cache track)
**Blocks:** 06, 07, 08
**Estimated effort:** 1–2 weeks

## Goal

The current runtime performs PG↔internal type conversions per-operation inside
the hot tuple loop. Lift those conversions to the loop's boundaries: decode
once at scan time into the JIT's preferred representation, operate in that
representation, encode once at output time. This is the foundation for specs
06–08 (per-type optimisations and bitcode inlining).

## Background

Today's per-tuple decode happens inside the JIT body via FFI calls into:
- `extract_field<T>` (`tuple_access.cpp:88`) → `heap_getattr` + `fromDatum<T>`
  on every field access.
- `get_int*_field_mlir`, `get_string_field`, `get_numeric_field`
  (`tuple_access.cpp:909-983`) — one C call per field per tuple.

Per-op conversions further down the pipeline:
- `numeric_to_i128(Datum, scale)` (`NumericConversion.cpp:99`) — called from
  `PostgreSQLRuntime.cpp:702` *during* aggregation/filtering, decoding the
  same value repeatedly.
- `i128_to_numeric(__int128, scale)` (`NumericConversion.cpp:148`) — called
  per-row during result encoding (`PostgreSQLRuntime.cpp:308`).
- Date conversions in `LowerToStd.cpp:140-152` — every date scanned multiplies
  by 86,400,000,000,000 to convert days→nanoseconds (and the reverse on output
  at line 194).
- `copy_datum_to_postgresql_memory` (`tuple_access.cpp:511`) — output encoding,
  per-column per-tuple.

The internal DB dialect representations are documented in
`include/lingodb/mlir/Dialect/DB/IR/DBOps.td`:

| DB type | Internal repr |
|---------|---------------|
| `db.date` | int64 nanoseconds since 2000-01-01 |
| `db.timestamp` | int64 nanoseconds |
| `db.decimal<p,s>` | __int128 scaled |
| `db.string` | VarLen32 (ptr + len) |

These are sensible *intermediate* representations for arithmetic. The waste
is in the decode/encode happening per-op rather than once per tuple per side.

## Design

### 1. Scan-time bulk decode

Add a "decoded tuple" representation that lives in stack/register state for
one tuple at a time:

```cpp
// generated MLIR equivalent — one struct per scan operator, columns it needs
struct DecodedTuple_<scan_id> {
    int64_t  col0_int;
    __int128 col1_decimal;
    int64_t  col2_date_nanos;     // already in DB dialect form
    VarLen32 col3_string;
    bool     col0_isnull, col1_isnull, ...;
};
```

The scan loop body decodes once at top of body, then column accesses become
struct field reads. The conversion cost moves out of *every* downstream op
into *one* op per column per tuple.

In MLIR terms, this is a new pattern in the DSA→Standard lowering
(`src/lingodb/mlir/Conversion/DSAToStd/`) that recognises a `dsa::ForOp`
over a `dsa::ScanSource` and emits the bulk decode at the top of the loop
body, replacing per-`dsa::At` conversions with the cached struct field reads.

### 2. New runtime entry point

Add to `src/pgx-lower/runtime/tuple_access.cpp`:

```cpp
extern "C" void decode_tuple_columns(
    int64_t iteration_signal,
    int32_t num_cols,
    const int32_t* col_indices,    // attribute numbers
    const int32_t* col_type_oids,  // for dispatch
    void* out_decoded_struct);     // typed by JIT
```

The body does one `heap_deform_tuple` (cheaper than N `heap_getattr` calls
because it walks the tuple once) and writes all columns into the output
struct, applying the same type conversions that currently happen per-op.

PostgreSQL's `heap_deform_tuple` is the right primitive — it's what
`slot_deform_tuple` uses internally, and it amortizes the variable-length
attribute walk.

### 3. Output-time bulk encode

Symmetric pattern at the result-streaming site
(`tuple_access.cpp:572-628`, `stream_tuple_to_destination`).

Today: `copy_datum_to_postgresql_memory` is called per-column-per-tuple,
each call doing a type-OID dispatch (`switch` at lines 517–553). Replace
with a per-result-shape encode function generated alongside the cached engine:

```cpp
extern "C" void encode_result_tuple(
    const DecodedResult* in,
    Datum* out_values,
    bool*  out_nulls);
```

The type-OID dispatch happens once, when the function is generated, not
once per tuple.

### 4. Numeric: keep __int128 across loop body

`numeric_to_i128` and `i128_to_numeric` are currently both called inside
output loops. After this spec:
- `numeric_to_i128` is called once per scanned tuple (in step 1's bulk decode).
- `i128_to_numeric` is called once per output tuple (in step 3's bulk encode).

Within the JIT body, numeric values stay as `__int128` — which they already
do; the FFI surface just becomes larger-grained.

## Files to touch

| File | Change |
|------|--------|
| `src/pgx-lower/runtime/tuple_access.cpp` | New `decode_tuple_columns`, new `encode_result_tuple` |
| `src/pgx-lower/runtime/tuple_access.cpp:88-114` | `extract_field<T>` becomes legacy fallback only — used when bulk decode doesn't apply (e.g., index scan paths) |
| `include/pgx-lower/runtime/tuple_access.h` | Declare new entry points |
| `src/lingodb/mlir/Conversion/DSAToStd/CollectionIterators.cpp:378` | Insert bulk-decode call at scan loop entry |
| `src/lingodb/mlir/Conversion/DSAToStd/` (new file `BulkDecodePattern.cpp`) | Pattern that rewrites per-op `dsa::At` to struct field reads |
| `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:296-308` | Switch `addDecimal` to use bulk encode path; keep old path as fallback |
| `src/pgx-lower/runtime/PostgreSQLRuntime.cpp:702` | Read decoded value from struct, not via `numeric_to_i128` |

## Don't change in this spec

- The internal DB dialect representations themselves (date=nanos, decimal=i128).
  Those are spec 06's territory.
- String representation. Spec 07.
- LLVM-level inlining. Spec 08.
- `extract_field<T>` and the per-field `get_*_field` functions stay — they're
  needed for non-bulk paths (index scans, single-row lookups). Bulk decode is
  an optimisation for sequential scan, the dominant TPC-H pattern.

## Acceptance criteria

- Build clean.
- All existing tests pass — correctness is the bar; this is a refactor.
- Unit test: a query like `SELECT l_quantity * l_extendedprice FROM lineitem
  WHERE l_shipdate > '1998-01-01' GROUP BY l_returnflag` exercises decimal,
  date, and string columns. Result must match PG bit-for-bit.
- Microbench (add a script, doesn't have to be the full TPC-H harness): time
  100k iterations of `extract_field<int64_t>` vs the bulk-decode equivalent.
  Bulk should be **≥3× faster** on the per-tuple path.
- A/B (warm, SF=0.01, 4-query subset):
  - **q01 (heavy aggregation): ≥15% faster.**
  - **q06 (scan + sum on lineitem.l_quantity): ≥10% faster.**
  - q03, q12: any direction acceptable; report the number.

## Risks

- The bulk decode pattern is only applicable when the JIT knows the column
  set up front. For dynamic patterns (rare in TPC-H), fall back to the old
  per-field path. Detect at lowering time, not runtime.
- `heap_deform_tuple` semantics around toasted columns: very large values
  are detoasted on access. Verify TPC-H doesn't trigger this at SF=0.01;
  document if it does.
- The MLIR pattern that rewrites `dsa::At` to struct reads is the trickiest
  part. If LingoDB's dialect doesn't expose enough handles, you may need a
  custom op in the pgx-lower side of the dialect tree. That's acceptable —
  this spec owns it.

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `05-decode`.

Required runs:
1. Standard 4-query sweep (cold + warm).
2. The microbench script for the per-tuple decode path.
3. Validation run: full 22-query TPC-H sweep at SF=0.01 must produce
   bit-identical results to baseline (use the harness's `result_validation`
   field — it hashes the first 5000 rows per query).

## Rollback

The bulk-decode pattern is gated by the new MLIR rewrite; if removed, the
per-field path is still in place. Revert the rewrite, runtime helpers stay
(harmless to keep). No data migration.
