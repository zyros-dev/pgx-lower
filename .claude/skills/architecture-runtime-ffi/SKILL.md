---
name: architecture-runtime-ffi
description: The C++ runtime called from JITed MLIR code. Hashtable, Vector, Sort, type conversions (date/numeric/string/varlena), tuple access globals, FFI symbols. Use when adding runtime functions, debugging JIT↔C boundaries, fixing memory leaks across the FFI, working on type conversions, or touching anything in src/pgx-lower/runtime/ or src/lingodb/runtime/.
---

# Runtime FFI surface

The JITed code calls back into C/C++ for everything that isn't pure
arithmetic. All runtime symbols are exported via `-Wl,--export-dynamic` so
LLVM's ExecutionEngine can resolve them at JIT time.

## File layout

```
src/pgx-lower/runtime/         (PG-specific)
├── PostgreSQLDataSource.cpp   PG heap scan adapter (DataSource impl)
├── PostgreSQLRuntime.cpp      TableBuilder, addDecimal, rt_set_execution_context
├── NumericConversion.cpp      numeric_to_i128, i128_to_numeric (NumericData parsing)
├── DateRuntime.cpp            extractYear/Month/Day, ExtractFromDate (nanos→PG date)
├── StringRuntime.cpp          like, concat, upper, lower, substring, fromInt, etc.
└── tuple_access.cpp           heap_getattr wrappers, get_*_field_mlir, result streaming

src/lingodb/runtime/            (generic, vendored)
├── Hashtable.cpp + LazyJoinHashtable.cpp    Chained hashtables (xxHash64)
├── Vector.cpp                  Doubling vector
├── Hash.cpp                    hashVarLenData (xxHash64)
├── PgSortRuntime.cpp           PG tuplesort wrapper
├── helpers.cpp + MetaData.cpp  + RuntimeSpecifications.cpp
└── PrintRuntime.cpp            Debug print (gate behind opt level)
```

Both directories are object libraries linked into `pgx_lower.so` with
`-export-dynamic`.

## Per-query global state (must reset every call)

The runtime is *stateful* across the FFI. The executor sets these before each
JIT call:

| Global | Defined in | Set by | Read by |
|--------|------------|--------|---------|
| `g_execution_context` | PostgreSQLRuntime.cpp:48 | rt_set_execution_context (called from JITEngine.execute, jit_execution_engine.cpp:135) | rt_get_execution_context |
| `g_tuple_streamer` | tuple_access.cpp:47 | my_executor.cpp:240-242 | result-streaming code |
| `g_current_tuple_passthrough` | tuple_access.cpp:48 | read_next_tuple_from_table | extract_field<T> |
| `g_computed_results` | tuple_access.cpp:45 | prepare_computed_results / store_*_result | stream_tuple_to_destination |
| `g_jit_table_oid` | tuple_access.cpp | open_postgres_table / query analyzer | (debug only) |
| `g_hashtable_metadata` | Hashtable.cpp:13 | Hashtable::create | Hashtable insert/destroy (carries spec + MemoryContext for deep copies) |

A spec that wants to cache compiled engines across queries (`specs/03`) must
reset all of these per call. The contract is "globals are query-scoped, the
JIT module is potentially process-scoped."

## Scan path (per-tuple decode)

`open_postgres_table(tableName)` (`tuple_access.cpp:347`):
- Calls `table_open` + `table_beginscan`.
- Sets `g_jit_table_oid`.
- Returns opaque handle.

`read_next_tuple_from_table(handle)` (`tuple_access.cpp:413`):
- Calls `table_scan_getnextslot`.
- Stores tuple in `g_current_tuple_passthrough.originalTuple` (line 461).
- Returns 1 if tuple, 0 at EOT, -1 on error.

`extract_field<T>(field_index, &is_null)` (`tuple_access.cpp:88`):
- `heap_getattr(g_current_tuple_passthrough.originalTuple, attr_num,
   tupleDesc, &isnull)` (line 107).
- Calls `fromDatum<T>(value, typeOid)` (line 114).
- **Called inside the per-tuple JIT loop** — this is the hot path.

Per-type FFI (called from JITed code):
- `get_int{16,32,64}_field`, `get_float{32,64}_field`, `get_bool_field`
- `get_string_field` (handles TEXTOID/VARCHAROID/BPCHAROID/NAMEOID + BPCHAR
  trim — see `tuple_access.cpp:768-829`)
- `get_numeric_field` (returns Numeric Datum, decoded later)
- MLIR wrappers: `get_int32_field_mlir(int64 sig, int32 idx)` etc. — the
  signal arg is a sentinel for the JIT.

## Output path (per-tuple encode)

`add_tuple_to_result(int64_t value)` (`tuple_access.cpp:717`):
- Triggers `process_computed_results_for_streaming`.

`copy_datum_to_postgresql_memory(Datum, Oid, isNull)` (`tuple_access.cpp:511`):
- Type-OID switch (lines 517-553).
- NUMERIC: `datumCopy`.
- TEXT/VARCHAR/BPCHAR: `datumCopy`.
- Numeric scalar types: returned as-is (Datum bit-pattern is correct).
- INTERVAL: converts int64 nanos → `Interval` struct (line 543-553).

`stream_tuple_to_destination(slot, dest, values, nulls, n)`
(`tuple_access.cpp:572`):
- Populates `slot->tts_values` and `tts_isnull`.
- `ExecStoreVirtualTuple(slot)`.
- `dest->receiveSlot(slot, dest)` — the actual handoff to the DestReceiver.

## Type conversions (the type-gap track)

These all sit in hot loops today. Specs 05–07 hoist them.

**Date/Timestamp** (`DateRuntime.cpp`):
- LingoDB internal repr is **i64 nanoseconds since 2000-01-01**.
- PG's DateADT is **i32 days since 2000-01-01**.
- `nanosToPostgresDate` (line 14): divides by 86_400_000_000_000.
- `postgresTimestampToNanos` (line 20): adds 946684800000000, multiplies by 1000.
- `extractYear/Month/Day(int64 nanos)`: convert to `DateADT`, call PG's `j2date`.
- **`addMonths` / `subtractMonths` are NOT IMPLEMENTED** (lines 26, 34 — `ereport(ERROR)`).

**Numeric** (`NumericConversion.cpp`):
- `numeric_to_i128(Datum, target_scale) -> __int128` (line 99): walks PG's
  variable-length Numeric (NumericShort vs NumericLong, base-10000 digits).
- `i128_to_numeric(__int128, scale) -> Datum` (line 148): allocates new
  Numeric varlena via `palloc`.
- Numerics cross the JIT boundary as **`__int128` scalars**, NOT varlena
  pointers. Decoded at scan, encoded at output.

**Strings** (`StringRuntime.cpp`):
- LingoDB internal repr is `VarLen32 = {uint8_t* ptr, uint32_t len}`.
- All ops take/return VarLen32 (length precomputed).
- `pgx_alloc()` (line 40): allocates from `CurrentMemoryContext` (PG side)
  or `new[]` (test mode).
- BPCHAR trim happens in `get_string_field` — eager strip of trailing spaces.
- Converters: `fromInt`, `fromFloat32/64`, `fromDecimal`, `fromChar`.
- Comparisons: `compareEq`, `compareLt`, etc.
- Pattern: `like`, `ilike`, `endsWith`, `startsWith`.

## Hashtables

`include/lingodb/runtime/Hashtable.h` + `LazyJoinHashtable.h`.

**Two flavors**:
- `Hashtable` — eager. Used by GROUP BY aggregation.
- `LazyJoinHashtable` — defers bucket allocation until finalize. Used by
  HashJoin build side.

**Layout**:
- `FixedSizedBuffer<Entry*> ht` — bucket array, sized `initialCapacity * 2`.
- `Vector values` — packed `{Entry* next, size_t hashValue, key..., value...}`.
- Hash function: xxHash64 via `hashVarLenData()` (Hash.cpp:4-6).
- Collision: chaining via `Entry::next`.
- Growth: bucket array doubles via `resize()` (Hashtable.cpp:198-220).

**FFI** (called from JITed code):
- `Hashtable::create(typeSize, initialCapacity, specPtr)` → `Hashtable*`
- `Hashtable::appendEntryWithDeepCopy(hash, currentLen, key, val, keySize, valSize)`
- `Hashtable::resize()`
- `Hashtable::destroy(ht)`
- LazyJoinHashtable variant has `finalize()` to build bucket index post-build.

**Metadata**: `g_hashtable_metadata` map (Hashtable.cpp:13) carries the
`HashtableSpecification` and PG `MemoryContext` for deep-copying VarLen32 strings
into the right context.

## Vector and GrowingBuffer

`runtime::Vector` (`Vector.h:6-29`):
- Single contiguous malloc'd buffer.
- Tracks `len`, `cap`, `typeSize`.
- Growth: doubles on `resize()`.
- Sort: `Vector::sort(compareFn)` — std::sort with user callback.

`runtime::GrowingBuffer` + `FlexibleBuffer`:
- Multi-segment growing buffer.
- Per-segment doubling.
- Used for parallel-friendly aggregation accumulators (infrastructure present
  but not exercised — backend is single-threaded).

## Sort runtime

`PgSortRuntime.cpp` (~650 lines).

Wraps PG's `tuplesort_begin_heap` — **uses PG's actual sort, not a custom
implementation**. Means we get PG's collations, nulls-first/last, and disk
spill via `work_mem` for free.

Flow:
- `PgSortState::create(tupleSize, specPtr)` — initializes tuplesort state.
- `PgSortState::appendTuple(data)` — packs MLIR tuple → Datum array via
  `unpack_mlir_to_datums` (line 187), feeds to tuplesort.
- `PgSortState::performSort()` — triggers `tuplesort_performsort`.
- `PgSortState::getNextTuple()` — returns sorted tuple, packed via
  `pack_datums_to_mlir` (line 278).

Decimal handling here was the source of multiple bugs (commits 081c1c4, ca39005,
1d46d60, 5295091). Watch for scale issues if you touch this.

## DataSource abstraction

`include/lingodb/runtime/DataSourceIteration.h:6-38`:

```cpp
class DataSource { virtual void* getNext() = 0; };
class DataSourceIteration {
    static DataSourceIteration* start(ExecutionContext*, VarLen32);
    bool isValid(); void next(); void access(RecordBatchInfo*);
    static void end(DataSourceIteration*);
};
```

`PostgreSQLDataSource.cpp` is the only implementation. Parses a JSON-ish
description (table name + OID) and calls `open_postgres_table`. Arrow
forward declarations exist (`include/lingodb/arrow/type_fwd.h`) but no
DataSource impl uses them.

## Adding a new runtime function

1. Implement in `src/pgx-lower/runtime/<file>.cpp` or
   `src/lingodb/runtime/<file>.cpp`.
2. Mark `extern "C"` if called from JIT (or rely on C++ name mangling +
   register the mangled name in `RuntimeFunctions.cpp`).
3. Confirm `-Wl,--export-dynamic` is present on the object library
   (it is — see `src/pgx-lower/runtime/CMakeLists.txt` and
   `src/lingodb/runtime/CMakeLists.txt`).
4. Add the function declaration to `db.runtime_call`'s registry if it should
   be callable from MLIR. Look in `src/lingodb/mlir/Conversion/DBToStd/` for
   the runtime-defs include and the `RuntimeFunctions` registration.
5. The JIT will resolve the symbol at compile time via dlsym in
   `mlir::ExecutionEngine`.

## Common gotchas

- **MemoryContext mismatch**: allocate with `CurrentMemoryContext` or PG's
  `palloc` will explode at free time. The executor switches context to
  `estate->es_query_cxt` before the JIT call; honor that.
- **VarLen32 lifetime**: scan-time strings live in PG's per-tuple context.
  If you store one (e.g., as a hashtable key), `datumCopy` it into the
  hashtable's owning context. See `Hashtable::appendEntryWithDeepCopy`.
- **Numeric precision**: numeric scale is in `typmod`, *not* in the Datum
  itself. The translator passes scale to runtime conversions; if you forget,
  you get wrong values silently.
- **PrintRuntime ops in hot loops**: emit actual LLVM IR calls. Don't leave
  them in committed lowerings. Commit 559d80e was a revert of exactly this.
- **Symbol export**: if a new runtime function fails JIT lookup, check that
  its object file has `-export-dynamic` and that no inline-only path
  (`static inline`, `static`) hides it.

## Related skills

- `architecture-mlir-dialects` — DSA's `dsa.ht_insert`, `dsa.lookup`, `dsa.sort`
  lower into the runtime functions documented here.
- `architecture-jit-compilation` — how symbols are resolved in the JIT.
- `architecture-execution-path` — when the per-query state gets reset.
