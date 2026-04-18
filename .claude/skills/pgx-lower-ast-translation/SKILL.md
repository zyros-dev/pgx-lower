---
name: pgx-lower-ast-translation
description: How PostgreSQL plan trees (PlannedStmt) are translated into MLIR RelAlg by PostgreSQLASTTranslator. Plan node coverage, expression translation dispatch, the TranslationContext, schema/type mapping. Use when adding plan node support, debugging "unsupported plan node type" errors, working on expression translation, or touching anything in src/pgx-lower/frontend/SQL/.
---

# AST translation (PG → MLIR RelAlg)

Lives in `src/pgx-lower/frontend/SQL/`. Entry point is
`PostgreSQLASTTranslator::translate_query(PlannedStmt*)` returning
`std::unique_ptr<mlir::ModuleOp>`.

## File layout

```
frontend/SQL/
├── postgresql_ast_translator.cpp   Public API + impl entry
├── query_analyzer.cpp              Pre-translation gate (see execution-path skill)
├── pgx_lower_constants.h           OID separators, magic strings
└── translation/
    ├── translation_core.cpp        translate_const, type mapping
    ├── translator_internals.h      Impl class, TranslationContext, helpers
    ├── translation_pch.h           Precompiled header — keep small!
    ├── schema_manager.cpp          Catalog access (table OIDs, column names, TupleDesc)
    ├── plan_translator_utils.cpp   Dispatcher + Sort/Limit/Gather/Material/Memoize
    ├── plan_translator_scans.cpp   SeqScan, IndexScan, IndexOnlyScan, Bitmap, Subquery, CTE
    ├── plan_translator_joins.cpp   HashJoin, MergeJoin, NestLoop (and Hash pass-through)
    ├── plan_translator_agg.cpp     Aggregation (GROUP BY, HAVING, all aggregate funcs)
    ├── expression_translator_basic.cpp      Var, Const, Aggref, Param + main dispatcher
    ├── expression_translator_complex.cpp    BoolExpr, NullTest, Coalesce, Case, CoerceViaIO, ScalarArrayOp
    ├── expression_translator_functions.cpp  FuncExpr, expression_for_stream (MAP wrapping)
    └── expression_translator_operators.cpp  OpExpr (arithmetic, comparison, LIKE, ||)
```

## Entry point

`postgresql_ast_translator.cpp:14-50`:

```cpp
auto translate_query(PlannedStmt* stmt) -> std::unique_ptr<mlir::ModuleOp> {
    auto module  = ModuleOp::create(...);          // line 29
    auto builder = OpBuilder(...);                  // line 30
    auto ctx     = QueryCtxT{builder, module, ...}; // line 32 — TranslationContext
    create_query_function();                        // line 34 — emits func.func
    generate_rel_alg_operations();                  // line 40 — walks plan tree
    create_materialize_op();                        // line 75-79 — wraps in MaterializeOp
    builder.create<func::ReturnOp>();               // line 76
    return module;
}
```

The MLIRContext is *passed in* — caller (mlir_runner.cpp) owns it. Returned
ModuleOp is owned by the caller.

## TranslationContext

`translator_internals.h:123-218`. Threaded through every translator call.

Fields:
- `current_stmt` — the PlannedStmt being translated.
- `builder` — single primary OpBuilder.
- `current_module`, `current_tuple`, `outer_tuple` — region-scoped values.
- `varno_resolution: map<(int,int), (str,str)>` — PG `(varno, varattno)` →
  `(table_scope, column_name)`. **Shared across child contexts**, populated by
  scans, queried by `translate_var()`.
- `params: map<int, ResolvedParam>` — NestLoop parameterization (correlated
  subqueries).
- `initplan_results: map<int, TranslationResult>` — InitPlan/CTE outputs.

Three child-context factories (lines 136-172):
1. `createChildContext(parent)` — same builder, inherits varno_resolution + params.
2. `createChildContext(parent, builder, tuple)` — region body with custom builder.
3. `createChildContextWithOuter(parent, outer_result)` — for correlated subqueries.

## Plan node coverage

Dispatcher: `translate_plan_node()` in `plan_translator_utils.cpp:61-99`,
switch on `nodeTag(plan)`.

| Node | File:Line | RelAlg op | Status |
|------|-----------|-----------|--------|
| T_SeqScan | scans.cpp:52-158 | BaseTableOp | ✓ |
| T_IndexScan | scans.cpp:160-271 | BaseTableOp | ✓ |
| T_IndexOnlyScan | scans.cpp:273-395 | BaseTableOp | ✓ |
| T_BitmapHeapScan | scans.cpp:397-508 | BaseTableOp | ✓ |
| T_SubqueryScan | scans.cpp:510-612 | (child) + Projection | ✓ |
| T_CteScan | scans.cpp:614-721 | (InitPlan) + Projection | ✓ |
| T_HashJoin | joins.cpp:133-197 | InnerJoin/Outer/Semi/AntiSemi (impl="hash") | ✓ |
| T_MergeJoin | joins.cpp:71-131 | InnerJoin/OuterJoin | ✓ |
| T_NestLoop | joins.cpp:211-346 | InnerJoin/OuterJoin + params | ✓ |
| T_Hash | joins.cpp:199-209 | (pass-through) | ⚠ |
| T_Agg | agg.cpp:160-633 | AggregationOp + post-MapOp | ✓ |
| T_Sort | utils.cpp:101-195 | SortOp | ✓ |
| T_Limit | utils.cpp:197-273 | LimitOp | ✓ |
| T_Gather, T_GatherMerge | utils.cpp:275-320 | (pass-through, no parallelism) | ⚠ |
| T_Material, T_Memoize | utils.cpp:325-342 | (pass-through) | ⚠ |
| T_Result, T_Group, T_Unique, T_SetOp | — | — | ✗ rejected |
| JOIN_FULL (full outer) | joins.cpp:733-735 | — | ✗ throws "not implemented" |

When you see "unsupported plan node type: %d" in logs, it's the default arm
of `translate_plan_node` (line 94).

## Expression translation

Dispatcher: `translate_expression()` in `expression_translator_basic.cpp:52-109`.

| Expr type | File | Notes |
|-----------|------|-------|
| T_Var | basic.cpp:111-208 | Uses varno_resolution + OUTER_VAR fallback |
| T_Const | basic.cpp:210-213 → translation_core.cpp:179-332 | Per-OID dispatch |
| T_Param | basic.cpp:277-313 | PARAM_EXEC only; reads from context.params |
| T_OpExpr | operators.cpp:54-165 | +,-,*,/,%, =,<,>,<=,>=,!=, LIKE, \|\| |
| T_FuncExpr | functions.cpp:170-200+ | Whitelist of supported functions |
| T_BoolExpr | complex.cpp:74-180 | AND/OR/NOT with NULL truth tables |
| T_NullTest | complex.cpp:182-200+ | IS NULL / IS NOT NULL |
| T_CoalesceExpr, T_CaseExpr, T_ScalarArrayOpExpr | complex.cpp | … |
| T_Aggref | basic.cpp:215-275 | Resolves via varno_resolution (set by Agg translator) |
| T_CoerceViaIO | complex.cpp:57-72 | Type cast via db.CastOp |
| T_RelabelType | basic.cpp:96-101 | Strips, recurses |

**Not supported**: T_WindowFunc, T_RowExpr, T_SubscriptingRef (array
slicing), JSON operators, recursive CTEs, lateral joins.

## Schema and type mapping

`schema_manager.cpp` + `translation_core.cpp:30-145`.

`PostgreSQLTypeMapper::map_postgre_sqltype` (`translation_core.cpp:30-77`):

| PG OID | MLIR type |
|--------|-----------|
| INT2/4/8OID | i16 / i32 / i64 |
| FLOAT4/8OID | f32 / f64 |
| BOOLOID | i1 |
| TEXTOID, VARCHAROID, BPCHAROID | db.string (TODO: BPCHAR should be db.char<N>) |
| NUMERICOID | db.decimal<p,s> (extracted from typmod) |
| DATEOID | db.date<day> |
| TIMESTAMPOID | db.timestamp<unit> (precision from typmod) |
| INTERVALOID | db.interval<daytime> |
| BYTEAOID | db.string |
| (anything else) | throws "Unknown PostgreSQL type OID" |

Nullable wrapping: `mlir::db::NullableType::get()` if column is nullable.

`get_table_oid_from_rte()` (`schema_manager.cpp:148`) extracts table OID from
the RTE. Table identifier format in BaseTableOp: `"<table_name>#<table_oid>"`
(see `pgx_lower_constants.h` for `TABLE_OID_SEPARATOR`).

## Idioms

- **Logging**: `PGX_LOG(AST_TRANSLATE, DEBUG, "...")` everywhere. `PGX_ERROR`
  + `throw std::runtime_error` for hard failures.
- **TranslationResult**: every plan-node translator returns one with `.op`,
  `.columns`, `.current_scope`, `.left_child_column_count`. Chain through.
- **Column manager**: scoped via
  `context.getOrLoadDialect<RelAlgDialect>()->getColumnManager()`. Use
  `getUniqueScope(alias)` once per scan; `createRef` and `createDef` produce
  the column attributes RelAlg ops need.
- **MapOp wrapping**: `translate_expression_for_stream` (in functions.cpp:57)
  wraps non-Var expressions in a MapOp with a generated computed column.

## Known TODOs and fragile spots

| Where | Issue |
|-------|-------|
| `translation_core.cpp:47-48` | BPCHAR maps to string, not db.char<N>. Length lost. |
| `translation_core.cpp:262` | INTERVAL is hardcoded daytime microseconds. |
| `plan_translator_agg.cpp:371` | AVG in combining mode (parallel agg) likely wrong. |
| `plan_translator_joins.cpp:733-735` | FULL OUTER JOIN throws. |
| `plan_translator_utils.cpp:720` | Column lookup in `apply_projection` uses raw lookup, not TranslationResult. |
| `expression_translator_basic.cpp:197` | "Safety check is goofy" — defensive but smelly. |
| `plan_translator_utils.cpp:298-320` | GatherMerge ignored — parallel plans become sequential. |

## Adding a new plan node

1. Add a `case T_NewNode:` to the dispatcher in
   `plan_translator_utils.cpp:61-99`.
2. Implement `translate_new_node()` in the appropriate `plan_translator_*.cpp`
   file (or a new one, register in CMake).
3. If it produces a new RelAlg op, that op must already exist in the dialect
   (`include/lingodb/mlir/Dialect/RelAlg/IR/RelAlgOps.td`).
4. Update `query_analyzer.cpp:151-233` (`analyzeNode`) so the analyzer
   doesn't reject queries containing it.
5. Add a regression test in `tests/sql/` and expected output in
   `tests/expected/`. Register in `extension/CMakeLists.txt:39-85`.

## Adding a new expression type

1. Add a `case T_NewExpr:` to the dispatcher in
   `expression_translator_basic.cpp:52-109`.
2. Implement in `expression_translator_complex.cpp` (most non-trivial cases
   live here).
3. If it needs a new DB op, add it to the DB dialect first
   (see `pgx-lower-mlir-dialects` skill).

## Related skills

- `pgx-lower-mlir-dialects` — what RelAlg ops exist; what DB/DSA/util look
  like downstream.
- `pgx-lower-execution-path` — what calls translate_query (the MLIR runner)
  and what happens after.
