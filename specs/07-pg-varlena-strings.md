# Spec 07 — Use PG varlena directly for strings

**Tier:** 2 (type-gap track)
**Stack on:** 05 (decode-at-scan)
**Blocks:** 08
**Estimated effort:** 1 week

## Goal

Drop the `VarLen32` (ptr+len) intermediate representation for strings/BPCHAR
and operate directly on PostgreSQL `varlena*` pointers. This removes a
per-string allocation in `fromXX` paths and eliminates the BPCHAR padding
"fudges" in `StringRuntime.cpp`.

## Background

Today the JIT operates on `VarLen32` (defined in `include/lingodb/runtime/`):

```cpp
struct VarLen32 { uint8_t* ptr; uint32_t len; };
```

PG's varlena is:

```c
struct varlena {
    char vl_len_[4];   // VARSIZE_ANY header (1B short or 4B long)
    char vl_dat[];
};
```

Conversion sites:
- Scan (`tuple_access.cpp:768-829`, `get_string_field`): extracts varlena,
  emits `VarLen32` for the JIT.
- Output (`StringRuntime.cpp:295-385`, `fromInt`/`fromFloat`/etc.): allocate
  new memory via `pgx_alloc`, return `VarLen32`.
- Round-trip in result encoding (`tuple_access.cpp:528-530`,
  `copy_datum_to_postgresql_memory`): copies varlena via `datumCopy`.

Most ops in `StringRuntime.cpp` (length, substring, compareEq, like, etc.)
take `VarLen32` and could just as easily take `varlena*`. The only saving in
the current setup is that `len` is precomputed; that's recoverable from
`VARSIZE_ANY_EXHDR(vl)` cheaply.

BPCHAR fudges: `get_string_field` with `BPCHAROID` strips trailing spaces
(`tuple_access.cpp:792-829`). PG's storage keeps them; pgx-lower trims them
to behave as if BPCHAR were CHAR. This is correct semantically but means
strings stored in BPCHAR columns get reallocated to a trimmed copy on every
scan. Direct varlena handling lets us trim lazily — only when comparing —
which avoids the alloc.

## Design

### 1. Replace `VarLen32` with `varlena*` at the JIT boundary

The DB dialect's `db.string` type's runtime representation becomes `i8*`
(pointing to the varlena header). Length is computed on demand via
`VARSIZE_ANY_EXHDR` macro, lowered as inline LLVM IR (it's a few bit ops on
the first byte).

`include/lingodb/mlir/Dialect/DB/IR/DBOps.td`: update the lowering of
`db::StringType` from "VarLen32 struct (ptr,len)" to "i8* (varlena pointer)".

### 2. Inline VARSIZE access

Add MLIR helpers (or inline IR in the DBToStd lowering) for:
- `db_string_length(str)` → load 1 byte, branch on the short/long header bit,
  return length.
- `db_string_data_ptr(str)` → conditional offset (1 or 4 bytes) from header.

These are 5–10 LLVM IR instructions inlined per access, cheaper than the
current `len` field load *and* avoiding the upfront precompute.

### 3. Rewrite StringRuntime to take varlena

`src/pgx-lower/runtime/StringRuntime.cpp`:
- All functions taking `VarLen32` (line 127, 130, 134, 153, 180, 207, 234,
  423, 430, 436, 447, 460, 475, 487, 500, 527, 531, 537, 564, 585, 607, ...)
  switch to `text*` parameters.
- `fromInt` / `fromFloat` / `fromDecimal` / `fromChar` (lines 295, 310, 325,
  340, 385): allocate via `cstring_to_text_with_len` (PG's standard helper)
  rather than `pgx_alloc` + manual VarLen32 build.
- `concat` / `concat3` (lines 447, 460): use `text_catenate`-equivalent.
- `upper` / `lower` (lines 475, 487): emit varlena directly.
- `substring` (line 500): use `text_substring`.

The exported FFI symbols stay the same names — only the parameter/return
types change. Since the JIT generates the calls, only the MLIR-side type
signatures need updating to match.

### 4. BPCHAR semantics

Stop eagerly trimming in `get_string_field` (`tuple_access.cpp:792-829`).
Instead:
- Comparison ops (`compareEq` et al.) detect BPCHAR via type metadata at
  *generation* time and emit a trim-aware compare.
- The decoded struct from spec 05 stores the raw varlena pointer plus a
  type-OID hint.

If this turns out to be invasive (BPCHAR semantics ripple through every
string op), a simpler v1 is: keep the eager trim but use varlena underneath.
Document the choice.

### 5. Memory ownership

PG's MemoryContext owns scan-time varlenas (they live in the per-tuple
context). New strings allocated by `fromXX` ops live in
`CurrentMemoryContext`, which must be the right query context — pgx-lower
already switches to `estate->es_query_cxt` (`my_executor.cpp:204`), so this
is correct. Document the rule.

## Files to touch

| File | Change |
|------|--------|
| `include/lingodb/mlir/Dialect/DB/IR/DBOps.td` | DBString runtime type: i8* not VarLen32 struct |
| `src/lingodb/mlir/Conversion/DBToStd/LowerToStd.cpp` | Lower string ops via inline VARSIZE; remove VarLen32 packing |
| `src/lingodb/runtime/VarLen32.h` (if it exists separately) | Mark as legacy; runtime keeps for non-string varlens |
| `src/pgx-lower/runtime/StringRuntime.cpp` | Rewrite signatures and bodies as above |
| `include/lingodb/runtime/StringRuntime.h` | Update signatures |
| `src/pgx-lower/runtime/tuple_access.cpp:768-829` | Stop trimming BPCHAR in scan |
| `src/pgx-lower/runtime/tuple_access.cpp:528-530` | Update output path if changed |
| Any consumer of `VarLen32` for strings | Migrate; keep VarLen32 for non-string FFI |

## Don't change

- Non-string `VarLen32` users (e.g., the `description` parameter to
  `PostgreSQLDataSource::createFromDescription` at
  `PostgreSQLDataSource.cpp:48`). VarLen32 isn't going away — only string
  uses move to varlena.
- The MLIR `db.string` type itself — only its lowered runtime representation
  changes.
- Output streaming via `dest->receiveSlot` — that already uses Datums, so
  varlena is what it expects.

## Acceptance criteria

- Build clean.
- All existing tests pass.
- New tests:
  - String comparison on BPCHAR column matches PG semantics for trailing-space
    handling.
  - `SELECT length(c_name)`, `SELECT upper(c_name)`, `SELECT c_name LIKE
    'CUST%'` from `customer` table all return correct results.
  - Memory leak check: 1M-row scan with `SELECT length(c_name)` shouldn't
    grow palloc'd memory unbounded between tuples (per-tuple memory context
    should reset).
- A/B (warm, SF=0.01, 4-query subset):
  - **q03 (joins on customer.c_mktsegment, varchar): ≥10% faster.**
  - q01 (no string ops): unchanged.
  - q06: unchanged.
  - q12 (l_shipmode comparisons, BPCHAR): **≥5% faster** (BPCHAR fudges removed).
- Validation: full 22-query sweep bit-identical.

## Risks

- BPCHAR trimming change may affect query results if any TPC-H query depends
  on the eager-trim behaviour. Run validation early.
- VARSIZE inline expansion adds 5–10 IR instructions per string access.
  In tight string-comparison loops this could be slower than the precomputed
  `len` field if the optimizer doesn't hoist. Verify with q12's IR dump.
- The runtime FFI symbol table (`RuntimeFunctions.cpp:203-210`) registers
  function signatures by C++ mangled name. Changing parameter types changes
  the mangling — update registrations or the JIT will fail symbol resolution.

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `07-string`.

Required runs:
1. Standard 4-query sweep (warm + cold).
2. Full 22-query validation.
3. Memory check: long-running query (`SELECT count(*) FROM lineitem WHERE
   l_comment LIKE '%special%'`) — process RSS should plateau, not grow
   linearly.

## Rollback

Type-system change. Revert the branch. No data migration — PG storage layout
is unchanged.
