# Spec 03 — Plan-shape compile cache

**Tier:** 1 (free wins)
**Stack on:** 01 (hoist MLIR/JIT state)
**Blocks:** 04 (cost-gate)
**Estimated effort:** 1 week

## Goal

Cache compiled `mlir::ExecutionEngine` instances keyed on a normalized plan
signature, so warm queries skip the entire MLIR pipeline (~250ms on Q20) and
LLVM codegen (~15ms+).

This is the largest single warm-query win available. The thesis breakdown for
Q20 (250ms compile + 929ms execution = 1.18s) implies caching alone removes
~21% of total wall time, and that's before the cached engine benefits from
spec 02's vectorization.

## Background

Today there is **no cross-query reuse**. Every query path:

1. Allocates a fresh `MLIRContext` (`mlir_runner.cpp:70`) — fixed by spec 01.
2. Runs the full RelAlg→DB→DSA→Standard→LLVM pipeline (six PassManagers,
   `mlir_runner_phases.cpp:35-207`).
3. Constructs a fresh `JITEngine` (`mlir_runner.cpp:152`) which builds an
   `mlir::ExecutionEngine`, runs LLVM optimization, and looks up symbols
   (`jit_execution_engine.cpp:69-114`).

The compiled function pointer (`main_fn_`) and the runtime context-setter
(`set_context_fn_`) are looked up via `engine_->lookup()`
(`jit_execution_engine.cpp:367-391`). These pointers stay valid as long as
the `ExecutionEngine` is alive.

There is no existing cache scaffolding. The only per-query caches in the
codebase are inside `TranslationContext` (`translator_internals.h:130-132`)
which is local to a single translation pass.

## Design

### Cache structure

Lives inside the `MLIRRuntime` singleton from spec 01:

```cpp
struct CachedQuery {
    std::unique_ptr<mlir::ExecutionEngine> engine;
    void* main_fn;
    void* set_context_fn;
    std::vector<int> selected_columns;   // for tuple desc reconstruction
    TupleDesc result_tuple_desc;         // PG-side, copied with refcount
    uint64_t hit_count;
};

class CompileCache {
    std::unordered_map<std::string, std::unique_ptr<CachedQuery>> entries_;
    size_t max_entries_ = 256;          // configurable via GUC
public:
    CachedQuery* lookup(const std::string& key);
    CachedQuery* insert(std::string key, std::unique_ptr<CachedQuery> entry);
    void invalidate_all();              // on catalog change (see below)
};
```

Bound the cache. 256 entries is a starting heuristic; expose
`pgx_lower.cache_max_entries` GUC for tuning.

### Cache key

Build the key in a new file:
`src/pgx-lower/frontend/SQL/plan_signature.cpp` exposing
`std::string compute_plan_signature(const PlannedStmt* stmt)`.

The signature must include everything that affects generated code:

1. **Plan tree shape**, depth-first walk:
   - `nodeTag(node)` for every node
   - For `T_SeqScan`: `scan->scanrelid` and the resolved `RangeTblEntry->relid`
   - For `T_IndexScan`: index OID, scan direction, key columns
   - For `T_HashJoin`/`T_NestLoop`/`T_MergeJoin`: join type, hashclauses count,
     join column types
   - For `T_Agg`: `numCols`, sort columns, `aggstrategy`
   - For `T_Sort`: number of sort keys, key types
2. **Targetlist column types and names** — `exprType((Node*)tle->expr)` for each.
   The result tuple desc shape is part of the contract.
3. **Qual structure** — for each qualifier, traverse the expression tree
   recording `nodeTag` and operator OIDs. **Do not include constant values** —
   constants are folded into immediates by the LLVM optimizer regardless.
   Operator OIDs are essential because `>` on int4 vs `>` on numeric is
   completely different code.
4. **Parameter types** — from `stmt->paramExecTypes` and any `Param` nodes
   encountered during traversal. Types only, not values.
5. **Catalog version stamp** — see invalidation section.

Hash with `std::hash<std::string>` initially; if collisions become a worry,
switch to xxHash. The key is small (typically <1KB) so the hash overhead is
negligible vs. the win.

**Do not include**: column names (only positions matter for codegen), comments,
whitespace, prepared-statement names. The same physical plan from
`SELECT a FROM t WHERE a > 5` and `SELECT a FROM t WHERE a > 10` should hit
the same cache entry.

### Lookup integration

In `MyCppExecutor::execute` (`my_executor.cpp:370-401`), after
`QueryAnalyzer::analyzePlan` succeeds:

```cpp
auto signature = compute_plan_signature(stmt);
auto* cached = rt.cache.lookup(signature);
if (cached) {
    // skip compilation — go straight to execution
    return execute_cached(cached, queryDesc, dest);
}
// fall through to normal compile path, then cache the result
```

Refactor `executeJITWithDestReceiver` (`mlir_runner.cpp:151-165`) to return
the compiled `JITEngine` instead of consuming it, so the caller can either
execute and discard (rare) or execute and cache (common).

### Catalog invalidation

This is the tricky bit. A cached engine references table OIDs, type OIDs,
and column layouts. If the user runs `ALTER TABLE` between queries, the cache
must invalidate.

Two acceptable approaches:

**(a) Pessimistic stamp (recommended for v1).** Use PostgreSQL's
`SharedInvalidMessageCounter` or, simpler, register a syscache invalidation
callback via `CacheRegisterRelcacheCallback(invalidate_all_cb, 0)` in
`_PG_init`. Any DDL on any relation flushes the entire cache. This is
heavy-handed but safe and trivial.

**(b) Per-relation tagging.** Each cache entry records which relation OIDs
it depends on. Invalidation walks entries and drops only those touching the
changed relation. Defer this to v2 — it requires plumbing OIDs out of the
plan walk and into the cache entry, plus the callback machinery.

Use (a) for this spec. File a follow-up issue for (b).

### Engine reuse semantics

`mlir::ExecutionEngine` is movable but not copyable. The cache owns it via
`unique_ptr`. Multiple sequential queries hitting the same entry simply call
the same function pointer; no per-call cloning needed.

The function pointer's behaviour depends on three globals reset per query:
- `g_execution_context` — set by `rt_set_execution_context(estate)` from
  `JITEngine::execute` (`jit_execution_engine.cpp:135-137`). Already correct.
- `g_tuple_streamer` — initialized per-query in `my_executor.cpp:240-242`.
  Already correct.
- `g_current_tuple_passthrough`, `g_computed_results` — per-query state owned
  by the executor. Already correct.

So long as the executor continues to wire these up before each call, the
cached engine is safe to reuse.

### Cache eviction

LRU on `hit_count` + last-used timestamp. When `entries_.size() == max_entries_`
on insert, evict the lowest-hit-count entry. Simple linear scan is fine for
N=256.

### GUCs to add

`executor_c.c:79-85` — add to `_PG_init`:

| GUC | Type | Default | Purpose |
|-----|------|---------|---------|
| `pgx_lower.cache_enabled` | bool | true | Master switch (for A/B testing this spec) |
| `pgx_lower.cache_max_entries` | int | 256 | LRU bound |
| `pgx_lower.cache_log_hits` | bool | false | Per-query hit/miss logging |

## Files to touch

| File | Change |
|------|--------|
| `include/pgx-lower/execution/mlir_runtime.h` (from spec 01) | Add `CompileCache cache;` member |
| `src/pgx-lower/execution/mlir_setup/mlir_runtime.cpp` (from spec 01) | Wire cache lifetime |
| `src/pgx-lower/execution/compile_cache.cpp` (new) | `CompileCache` impl, syscache callback |
| `include/pgx-lower/execution/compile_cache.h` (new) | Public API |
| `src/pgx-lower/frontend/SQL/plan_signature.cpp` (new) | `compute_plan_signature` |
| `include/pgx-lower/frontend/SQL/plan_signature.h` (new) | Signature API |
| `src/pgx-lower/execution/postgres/my_executor.cpp:370-401` | Cache lookup before compile |
| `src/pgx-lower/execution/mlir_runner.cpp:151-165` | Return engine instead of consuming |
| `src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp` | Add `release_engine()` and `attach_engine()` for cache reuse |
| `src/pgx-lower/execution/postgres/executor_c.c:79-85` | Register new GUCs and `CacheRegisterRelcacheCallback` |
| `extension/sql/pgx_lower--1.0.sql` | Optionally expose `pgx_lower_cache_stats()` SRF for visibility |

## Acceptance criteria

- Build clean.
- All existing tests pass.
- New unit tests:
  - `compute_plan_signature` returns identical strings for identical plans
    with different constants.
  - `compute_plan_signature` returns different strings when the plan shape,
    qual operator, or column types differ.
  - `CompileCache` evicts when over capacity.
  - Syscache callback flushes the cache (test by issuing `ALTER TABLE` then
    verifying next query missed).
- A/B (warm, SF=0.01, 4-query subset, 5 iterations):
  - **Warm queries (iter 2+) ≥30% faster** on q01 and q03 (heavy plans,
    biggest cache benefit). Iter 1 should match baseline.
  - q06 may not improve much (compile cost is small there) but must not regress.
- A/B with `pgx_lower.cache_enabled = off`: numbers must match baseline
  (sanity check that the cache didn't subtly break the cold path).
- DDL test: `ALTER TABLE lineitem ADD COLUMN x int; ALTER TABLE lineitem DROP COLUMN x;`
  between iterations forces a re-compile (verify via cache_log_hits).

## Risks

- **Stale plans across catalog changes.** The pessimistic invalidation is
  conservative; if it ever misses a callback, queries could execute against
  the wrong table layout. Test this carefully.
- **Memory growth.** 256 cached engines × multiple MB each = potentially
  hundreds of MB. Monitor RSS during the test suite.
- **Plan signature collisions.** A bug in `compute_plan_signature` that
  produces the same key for two semantically-different plans means the wrong
  code runs. Add an assertion comparing the cached `result_tuple_desc` against
  the new query's expected desc; mismatch → invalidate that entry and recompile.
- **Hash bias on `std::hash<std::string>`.** Acceptable for v1. If profiling
  shows the hash bucket walk is hot, switch to a faster hasher.

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `03-cache`.

Required runs:
1. Cache-enabled, 4-query subset, 5 iterations (warm dominates).
2. Cache-disabled (`pgx_lower.cache_enabled = off`), 4-query subset, 5
   iterations — must match baseline.
3. Catalog-invalidation test:
   - Run 4-query sweep, capture hit count.
   - Issue `ANALYZE lineitem` (forces invalidation under approach (a)).
   - Re-run 4-query sweep, hit count should restart from zero, then climb.

## Rollback

`SET pgx_lower.cache_enabled = false` disables the cache instantly. Code
revert is contained — the cache lookup is a single block in
`MyCppExecutor::execute`.
