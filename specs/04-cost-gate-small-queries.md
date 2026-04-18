# Spec 04 — Cost-gate small queries to PG interpreter

**Tier:** 1 (free wins)
**Stack on:** 03 (cache)
**Blocks:** —
**Estimated effort:** 2–3 days

## Goal

Skip pgx-lower compilation for plans the planner thinks are cheap. PostgreSQL's
interpreted executor wins on tiny queries (per-row interpretation overhead is
small; compilation overhead is large), and even with caching (spec 03) the
first execution of a cold plan pays full compile cost.

## Background

`QueryAnalyzer::isMLIRCompatible` (`query_analyzer.cpp:33-83`) is the current
gate. It's purely structural — checks query shape and column types. **No cost
inspection happens anywhere in pgx-lower today.**

PostgreSQL provides cost on the Plan struct:
- `Plan->startup_cost` (double)
- `Plan->total_cost` (double)
- `Plan->rows` (estimated row count)

Available at `stmt->planTree->total_cost` from any `PlannedStmt*`.

The fallback path is already wired: when `MyCppExecutor::execute` returns
false, `custom_executor` (`executor_c.c:41-48`) calls `prev_ExecutorRun_hook`
or `standard_ExecutorRun`. Cost-gating just adds another reason to return false.

## What to change

### 1. New GUCs

`src/pgx-lower/execution/postgres/executor_c.c:79-85` — add:

```c
DefineCustomRealVariable("pgx_lower.min_total_cost",
    "Skip pgx-lower for plans below this estimated total_cost",
    NULL, &g_min_total_cost,
    100.0,         // default — covers most "single-row lookup" patterns
    0.0,           // min — 0 disables the gate
    1e12,          // max
    PGC_USERSET, 0, NULL, NULL, NULL);

DefineCustomRealVariable("pgx_lower.min_rows",
    "Skip pgx-lower for plans estimating fewer rows than this",
    NULL, &g_min_rows,
    100.0, 0.0, 1e12,
    PGC_USERSET, 0, NULL, NULL, NULL);
```

The `100.0` default is a placeholder. Calibrate it with the A/B sweep — see
"Acceptance criteria" below.

### 2. Cost check in QueryAnalyzer

Extend `QueryCapabilities` (`include/pgx-lower/frontend/SQL/query_analyzer.h:18-33`):

```cpp
struct QueryCapabilities {
    // existing fields...
    double plan_total_cost = 0.0;
    double plan_rows = 0.0;
    bool   below_cost_threshold = false;
};
```

In `QueryAnalyzer::analyzePlan` (`query_analyzer.cpp:125-149`), after the
`stmt->planTree` null check:

```cpp
caps.plan_total_cost = stmt->planTree->total_cost;
caps.plan_rows = stmt->planTree->plan_rows;
caps.below_cost_threshold =
    (caps.plan_total_cost < g_min_total_cost) ||
    (caps.plan_rows       < g_min_rows);
```

In `QueryCapabilities::isMLIRCompatible` (`query_analyzer.cpp:33-83`), add:

```cpp
if (below_cost_threshold) return false;
```

### 3. Logging

When the gate rejects a query, log via `PGX_LOG(GENERAL, DEBUG, ...)` with the
cost and threshold. This is essential for tuning the threshold per workload.

### 4. Interaction with the cache (spec 03)

Order of decisions in `MyCppExecutor::execute`:

1. Run `QueryAnalyzer::analyzePlan` — produces caps.
2. If `caps.isMLIRCompatible() == false` → return false (PG interpreter takes over).
3. Compute plan signature.
4. **Cache lookup** — if hit, run cached engine regardless of cost. (The
   compile cost is already sunk; running the cached fast code is free.)
5. Cache miss → compile and execute.

Step 4 is critical: cost-gating only avoids *first compilations*. Once a plan
is cached, the cost of running it is just the function call. **Do not bypass
the cache lookup based on cost.**

### 5. Don't gate the analyze-only path

If a future spec adds an `EXPLAIN`-style "would pgx compile this" probe, it
must bypass the cost gate. Not relevant now; flag it in the code with a
comment.

## Files to touch

| File | Change |
|------|--------|
| `src/pgx-lower/execution/postgres/executor_c.c:79-85` | Two new GUCs |
| `include/pgx-lower/frontend/SQL/query_analyzer.h:18-33` | Add cost fields |
| `src/pgx-lower/frontend/SQL/query_analyzer.cpp:33-83` | Cost rejection in `isMLIRCompatible` |
| `src/pgx-lower/frontend/SQL/query_analyzer.cpp:125-149` | Populate cost in `analyzePlan` |
| `src/pgx-lower/execution/postgres/my_executor.cpp:370-401` | Move cache lookup before cost gate (see step 4) |

## Acceptance criteria

- Build clean.
- Existing tests pass.
- Unit test: `analyzePlan` populates `plan_total_cost` correctly for a
  hand-built `PlannedStmt`.
- A/B (cold + warm, SF=0.01, 4-query subset):
  - **Cold q06: ≥20% faster** (q06 is the smallest query — the canonical
    cost-gate winner if our default threshold is right).
  - **Warm q01, q03 (cached after first execution): unchanged or better** —
    the cache must still hit on iteration 2+.
- Calibration sweep — required deliverable in PR:
  - Run the 4-query sweep at `min_total_cost` ∈ {0, 10, 100, 1000, 10000}.
  - Plot total wall time per setting (cold + warm bars).
  - Recommend the threshold that gives the best aggregate. Default the GUC
    to that value.
- Per-query reporting: the PR description must say "qN was below threshold X
  and routed to PG interpreter" for any query where that happened.

## Risks

- PG planner cost estimates are notoriously bad on small tables. SF=0.01 will
  exaggerate this — every table is tiny so total_cost is small for everything.
  The default threshold may need to be much smaller than would work at SF=10.
  Document this clearly: **the right `min_total_cost` value is workload-dependent.**
- A query whose first execution is cost-gated will never enter the cache.
  That's fine — if it's cheap enough, it should never be compiled.

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `04-gate`.

Special: this spec needs the calibration sweep above, not just a single A/B.
Add `--label "04-gate-threshold-${value}"` to each run for easy SQL grouping.

## Rollback

`SET pgx_lower.min_total_cost = 0` disables the gate. Code revert is small.
