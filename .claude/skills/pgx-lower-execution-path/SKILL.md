---
name: pgx-lower-execution-path
description: The full call chain from PostgreSQL ExecutorRun_hook through pgx-lower's dispatcher, query analyzer, MLIR runner, and JIT engine. Use when working on query dispatch, the executor hook, the fallback to PG's interpreter, per-query lifecycle, or any code in src/pgx-lower/execution/postgres/ or mlir_runner.cpp.
---

# pgx-lower execution path

The path a query takes from PG calling our hook to the JITed function returning
results. Most performance work in this codebase lives along this chain.

## The chain (file:line)

```
PostgreSQL standard_ExecutorRun()
  ↓ (via ExecutorRun_hook)
custom_executor()                      executor_c.c:35-49
  ↓
try_cpp_executor_internal()            executor_c.c:30-33
  ↓
try_cpp_executor_direct()              executor_c.cpp:147-160
  ↓
MyCppExecutor::execute(QueryDesc*)     my_executor.cpp:370-401
  ↓ (decides: compile or fall back?)
QueryAnalyzer::analyzePlan(stmt)       query_analyzer.cpp:125-149
  ↓ (if compatible)
run_mlir_with_ast_translation()        my_executor.cpp:346-368
  ↓
mlir_runner::run_mlir_with_dest_receiver()  mlir_runner.cpp:60-148
  ↓ (MLIR pipeline)
runCompleteLoweringPipeline()          mlir_runner_phases.cpp (phases 3a/3b/3c)
  ↓
executeJITWithDestReceiver()           mlir_runner.cpp:151-165
  ↓
JITEngine::compile() + execute()       jit_execution_engine.cpp:69-182
  ↓
[JITed code runs, calls runtime FFI, streams to DestReceiver]
```

## Hook installation

`_PG_init` in `executor_c.c:69-90` is called when PG loads `pgx_lower.so`:

1. Saves the previous `ExecutorRun_hook` into `prev_ExecutorRun_hook`.
2. Installs `custom_executor` as the new hook.
3. Registers GUCs (logging only — no perf knobs yet).
4. Calls `initialize_mlir_passes()` (`mlir_runner_passes.cpp:31-48`) which
   runs `mlir::registerAllPasses()` plus our custom passes. Once-only.

`_PG_fini` (`executor_c.c:92-95`) restores the previous hook.

## The decision: compile or fall back?

`QueryCapabilities::isMLIRCompatible()` (`query_analyzer.cpp:33-83`) is the
gate. Required conditions:

- `commandType == CMD_SELECT` (no INSERT/UPDATE/DELETE).
- All output column types in `isTypeSupportedByMLIR()` whitelist
  (`query_analyzer.cpp:313-331`): INT2/4/8, FLOAT4/8, BOOL, TEXT, VARCHAR,
  BPCHAR, NUMERIC, DATE, TIMESTAMP, INTERVAL.
- At least one of `requiresSeqScan`, `requiresAggregation`, `requiresJoin`,
  `requiresLimit` is true (i.e. there's *some* table access).
- All `FuncExpr` calls in the targetlist match a small whitelist (upper, lower,
  substring, casting funcs).

**Notable rejection patterns**:
- `SELECT 1` (no table access) → rejected.
- Index scans are detected as `T_IndexScan` and treated as `requiresSeqScan` —
  *accepted*, but the AST translator's index-scan path is a separate code path.
- Subqueries (`T_SubqueryScan`) recursively analyze the subplan.
- Unknown plan node types are *accepted* by default for testing
  (`analyzeNode:151-233` falls through). This means trouble queries may reach
  the translator and fail there.

**No cost-gating today**. `Plan->total_cost` is available but not inspected.
See `specs/04-cost-gate-small-queries.md`.

## Fallback path

If `MyCppExecutor::execute` returns `false`:

```c
if (!mlir_handled) {                              // executor_c.c:41-48
    if (prev_ExecutorRun_hook) {
        prev_ExecutorRun_hook(...);
    } else {
        standard_ExecutorRun(...);
    }
}
```

So pgx-lower never *blocks* PG; it can only *augment*. Good defensive design.

## Per-query setup (in MyCppExecutor)

Before invoking the MLIR runner:

1. `validateAndPrepareQuery()` extracts `PlannedStmt` from `QueryDesc`
   (`my_executor.cpp:295-304`).
2. `setupExecution()` creates `EState`, `ExprContext`, switches to
   `estate->es_query_cxt` MemoryContext (`my_executor.cpp:315-324`).
3. `setupResultProcessing()` reads targetlist, builds `TupleDesc`, allocates
   `TupleTableSlot`, initializes `g_tuple_streamer`
   (`my_executor.cpp:215-245`).
4. PG_TRY/PG_CATCH wrapper (`my_executor.cpp:326-344`) catches PG ereport
   inside MLIR translation.

## Per-query state in the runtime (must reset every call)

Globals living in `runtime/` that the JITed code reads/writes:

- `g_execution_context` (`PostgreSQLRuntime.cpp:48`) — set by JITEngine via
  `rt_set_execution_context(estate)` immediately before invoking `main`.
- `g_tuple_streamer` (`tuple_access.cpp:47`) — initialized in `my_executor.cpp`
  before each call, shut down after.
- `g_current_tuple_passthrough` (`tuple_access.cpp`) — set by
  `read_next_tuple_from_table`.
- `g_computed_results` — output staging.

Any spec that wants to cache compiled engines across queries (specs/03) must
keep this contract: globals reset per call, even if the engine is reused.

## What is fresh per query (and shouldn't be — see specs/01)

`mlir_runner.cpp:60-148`:
- Line 70: `mlir::MLIRContext context;` — stack-allocated per query.
- Line 71: `setupMLIRContextForJIT(context)` — re-loads dialects every time.
- Line 76: AST translator constructed.
- Line 152: `JITEngine engine(...)` — per-query temporary.

`mlir_runner_phases.cpp` lines 45, 62, 98, 117, 136, 166: six `PassManager`
constructions per query.

`jit_execution_engine.cpp:286-303`: fresh `TargetMachine` every JIT compile.

## PG_TRY semantics

PG uses `setjmp/longjmp` for exception handling. C++ exceptions thrown across
PG_TRY boundaries are *undefined behavior*. The codebase wraps the MLIR
pipeline in PG_TRY and converts internal `std::exception` to `ereport` before
re-throwing. See `mlir_runner.cpp:99-121`.

When working in this area: never throw a C++ exception across a PG_TRY without
catching first. Never call `ereport(ERROR, ...)` from inside a `try` block
without thinking about cleanup.

## Common gotchas

- **Memory contexts.** Switch to `estate->es_query_cxt` for any `palloc` that
  should live as long as the query. `CurrentMemoryContext` is global and
  changes on you. JITEngine saves/restores it around the JIT call.
- **The PG executor calls our hook *per call to ExecutorRun*, not per query.**
  `count > 0` may iterate; `execute_once == false` means partial fetch
  semantics. Today we treat each call as full execution; this is fine for
  TPC-H but breaks cursors.
- **`g_tuple_streamer.slot` will be valid only if we initialized it.** A
  cached engine that runs on a slot from query A while query B is in flight
  is a use-after-free. Spec 03's cache must enforce this.
- **The fallback isn't free.** `prev_ExecutorRun_hook` may be from another
  extension. Don't assume `standard_ExecutorRun` is the next link.

## Related skills

- `pgx-lower-ast-translation` — what `PostgreSQLASTTranslator::translate_query`
  does after `mlir_runner` calls it.
- `pgx-lower-jit-compilation` — what `JITEngine::compile` and `execute` do.
- `pgx-lower-runtime-ffi` — what `g_*` globals carry and what symbols the JIT
  calls back into.
