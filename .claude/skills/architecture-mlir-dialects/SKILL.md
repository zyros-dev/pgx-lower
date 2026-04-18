---
name: pgx-lower-mlir-dialects
description: The four MLIR dialects (RelAlg, DB, DSA, util) used by pgx-lower, their key types and ops, and the lowering pipeline (RelAlg → DB+DSA+Util → Standard → LLVM). Use when adding ops, debugging lowering passes, working on dialect types, touching anything in src/lingodb/mlir/, or wondering where a particular op lowers.
---

# MLIR dialects and lowering pipeline

pgx-lower vendors a snapshot of LingoDB's dialects (~432KB headers + sources).
Modern LingoDB has SubOp; we don't. Our pipeline:

```
RelAlg                   (high-level relational algebra)
  ↓ Phase 3a: RelAlgToDB Translator pattern
DB + DSA + util          (imperative DB ops, collections, refs)
  ↓ Phase 3b: DBToStd, DSAToStd ConversionPatterns
Standard (arith, scf, memref, cf, func) + util
  ↓ Phase 3c: StandardToLLVM, UtilToLLVM
LLVM dialect             (translatable to LLVM IR)
  ↓
LLVM IR + JIT compile
```

Phase orchestration: `mlir_runner_phases.cpp` (3a:35-86, 3b:88-150, 3c:152-207).

## Dialect 1: RelAlg

**Purpose**: Logical query plan. SQL-like operators on tuple streams.

**Key types** (`include/lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.td`):
- `TupleStream` (line 51) — unbounded sequence of tuples flowing through plan.
- `Tuple` (line 55) — single record.

**Load-bearing ops**:
| Op | Line | Emitted by | Lowered by |
|----|------|------------|------------|
| BaseTableOp | 153 | scans translator | BaseTableOp.cpp Translator |
| SelectionOp | 171 | filter translator + optimizer | SelectionOp.cpp |
| MapOp | 186 | expression-for-stream | MapOp.cpp |
| AggregationOp | 244 | agg translator | AggregationOp.cpp |
| SortOp | 259 | sort translator | SortOp.cpp |
| LimitOp | 199 | limit translator | LimitOp.cpp |
| InnerJoinOp/SemiJoinOp/AntiSemiJoinOp/OuterJoinOp | 280-298 | join translators | Joins.cpp + HashJoin/NLJoin |
| MaterializeOp | (in cpp) | top-level wrapper | LowerToDB.cpp pass — the entry hook |
| ConstRelationOp | 141 | inline constants | ConstRelationOp.cpp |

**Query optimization passes** (run before lowering, in
`src/lingodb/mlir/Dialect/RelAlg/Passes.cpp:13-30`): SimplifyAggregations,
ExtractNestedOperators, CSE, Canonicalize, DecomposeLambdas, Pushdown,
Unnesting, ReduceGroupByKeys, ExpandTransitiveEqualities, OptimizeJoinOrder,
CombinePredicates, OptimizeImplementations, IntroduceTmp.

## Dialect 2: DB

**Purpose**: Imperative scalar ops with NULL semantics and PG-aware types.

**Key types** (`include/lingodb/mlir/Dialect/DB/IR/DBOps.td`):
| Type | Line | Internal repr | Lowers to |
|------|------|---------------|-----------|
| NullableType<T> | 48 | `{i1 valid, T value}` | LLVM struct |
| DateType<unit> | 78 | i64 nanoseconds (always) | i64 |
| TimestampType<unit> | 104 | i64 nanoseconds | i64 |
| IntervalType<unit> | 90 | i64 nanos (daytime) or i32 (months) | i64 / i32 |
| DecimalType<p,s> | 110 | i128 scaled fixed-point | i128 |
| StringType | 120 | VarLen32 (i8* + i32 len) | LLVM struct |
| CharType<N> | 62 | i8 array | LLVM array |

**The nanosecond date repr is wasteful** — see `specs/06-pg-native-date-repr.md`.

**Load-bearing ops** (most have nullable variants via NullableType):
| Op | Purpose |
|----|---------|
| db.constant | Scalar constant |
| db.null | NULL value |
| db.isnull / db.nullable_get_val / db.as_nullable | Null bit manipulation |
| db.compare | =, !=, <, >, <=, >= (with NULL semantics) |
| db.add/sub/mul/div/mod | Arithmetic with type inference + decimal scaling |
| db.and/or/not | Logical with three-valued (true/false/NULL) semantics |
| db.cast | Conversions: int↔float, int↔decimal, int↔string, date variants |
| db.hash | Hash for join/agg |
| db.between/oneof | Range/set membership |
| db.runtime_call | FFI call to runtime (string ops, date arithmetic) |

## Dialect 3: DSA (Data Structure Abstraction)

**Purpose**: Collections, iteration, hashtables, builders. Where "what to do"
becomes "how to do it imperatively."

**Key types** (`include/lingodb/mlir/Dialect/DSA/IR/DSAOps.td`):
| Type | Line | Lowers to runtime |
|------|------|-------------------|
| VectorType<T> | 77 | runtime::Vector |
| RecordBatchType<T> | 87 | Arrow batch / row vector |
| JoinHashtableType<K,V> | 97 | runtime::Hashtable or LazyJoinHashtable |
| AggregationHashtableType<K,V> | 92 | runtime::Hashtable |
| TableBuilderType<T> | 64 | Vector + row buffer |
| TableType | 59 | runtime::Vector or Arrow table |
| FlagType | 48 | i1 alloca (loop control) |
| GenericIterableType<T, name> | 42 | parameterized iterator (e.g., "join_ht_iterator") |

**Load-bearing ops**:
| Op | Line | Purpose |
|----|------|---------|
| dsa.create_ds | 120 | Allocate Vector/Hashtable/TableBuilder |
| dsa.for(coll, until?) { region } | 210 | Iterate, optional early-stop flag |
| dsa.lookup(ht, key) | 192 | Hashtable probe → match iterator |
| dsa.ht_insert(ht, key, val, hash_region, equal_region, reduce_region) | 157 | Insert with custom hash/eq/reduce |
| dsa.ds_append | 175 | Append to vector/builder |
| dsa.finalize | 164 | Finalize hashtable/builder |
| dsa.at(record, pos) | 142 | Field access |
| dsa.scan_source(desc) | 186 | Open external data source — calls rt::DataSourceIteration::start |
| dsa.sort | 201 | In-place sort |
| dsa.{create,set,get}flag | 103-118 | Boolean flag for loop control |
| dsa.yield / dsa.cond_skip | 247-258 | Loop body terminators |

## Dialect 4: util

**Purpose**: Cross-cutting utilities. Pointer-like types, allocation, hashing,
varlen.

**Key types** (`include/lingodb/mlir/Dialect/util/UtilOps.td`):
- `RefType<T>` (35) — opaque pointer-to-T → llvm.ptr.
- `VarLen32Type` (51) — `{i32 length, i8* ptr}` for strings.

**Key ops**:
| Op | Purpose |
|----|---------|
| util.alloc / util.alloca | malloc / stack alloc |
| util.varlen32_create / _create_const | String descriptor / global constant |
| util.hash_64 / util.hash_varlen / util.hash_combine | xxHash-based hashing |
| util.PackOp / util.UnPackOp | Pack/unpack tuple from struct |

## The Translator pattern (RelAlg → DB+DSA+Util)

`src/lingodb/mlir/Conversion/RelAlgToDB/LowerToDB.cpp` and the `Translators/`
subdir. **Not** a regular MLIR ConversionPattern — Volcano-style code generation:

- Each RelAlg op gets a Translator class.
- `Translator::produce()` emits code to generate tuples.
- `Translator::consume(child)` receives tuples from a child.
- Translators chain top-down (RelAlg op order); code generates bottom-up
  (execution order).
- They're stateful — they hold hashtables, vectors, current scope.

Per-op translator files in `src/lingodb/mlir/Conversion/RelAlgToDB/Translators/`:
BaseTableOp, Selection, Map, Aggregation (largest), Joins (delegates to
HashJoinTranslator or NLJoinTranslator), Sort, Limit, ConstRelation,
SetOps, Renaming, Projection, Materialize, Tmp.

The pass entry point is `LowerToDBPass` (created via
`mlir::relalg::createLowerToDBPass()`) — runs at FuncOp granularity, finds
MaterializeOp hooks, dispatches to translators.

## Phase 3b: DB → Standard, DSA → Standard

`src/lingodb/mlir/Conversion/DBToStd/LowerToStd.cpp` (DB):
- Null handling: NullableType → struct
- Arith with overflow checks for decimals
- Comparison with NULL three-valued logic
- String ops via runtime calls (StringRuntime)
- Decimal: i128 ops with rescaling
- Date/Timestamp: nanosecond multipliers (lines 140, 142, 152, 194, 206 — see
  spec 06)

`src/lingodb/mlir/Conversion/DSAToStd/`:
- `DSAToStdPatterns.cpp` — main DSA ops
- `CollectionsToStdPatterns.cpp` — collection-specific
- `CollectionIterators.cpp` — iteration protocols (pgsort_iterator,
  join_ht_iterator, scan_source_iterator)
- `ScalarDSAToStdPatterns.cpp` — scalar conversions inside collections

## Phase 3c: Standard → LLVM

`src/lingodb/mlir/Conversion/StandardToLLVM/StandardToLLVM.cpp:72-100`:
- AffineToStd
- SCFToControlFlow
- UtilToLLVM (delegated)
- FuncToLLVM
- ControlFlowToLLVM
- FinalizeMemRefToLLVM
- ArithToLLVM

After phase 3c, all ops are LLVM dialect.

## Vector dialect status

**Not registered. Not used.** No code anywhere. See `specs/09-mlir-vector-dialect.md`
for adding it.

## SubOp dialect status

**Not present.** Modern LingoDB has it as the batching/vectorisation layer
between RelAlg and DB+DSA. We don't. See `specs/11-lingodb-rebase-decision.md`.

## Where to add a new dialect / op / pass

- **New op in existing dialect**: edit the `.td` file in
  `include/lingodb/mlir/Dialect/<Name>/IR/<Name>Ops.td`. TableGen regenerates
  on build (target: `PGXLower<Name>OpsIncGen`).
- **New lowering pattern**: add a file under
  `src/lingodb/mlir/Conversion/<From>To<To>/`, register in CMakeLists.
- **New pass**: add to `src/lingodb/mlir/`, register via `Passes.cpp` and
  `mlir::registerAllPasses()` in `initialize_mlir_passes`.
- **New top-level pipeline phase**: edit `mlir_runner_phases.cpp` to add a
  phase 3d (or modify 3a-c).

## Related skills

- `pgx-lower-runtime-ffi` — what `dsa.scan_source`, `dsa.ht_insert` etc. lower
  into at runtime (the C++ Hashtable, Vector, etc.).
- `pgx-lower-jit-compilation` — what happens to the LLVM dialect output.
- `pgx-lower-ast-translation` — what produces RelAlg ops in the first place.
