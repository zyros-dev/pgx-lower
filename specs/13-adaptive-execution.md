# Spec 13 — Two-tier adaptive execution (deferred / research)

**Tier:** 4 (research)
**Stack on:** 09 landed
**Blocks:** —
**Estimated effort:** 6–8 weeks (research-grade)

## Goal

Umbra-style adaptive execution: start a query with a cheap interpreter (or a
lightweight JIT like spec 12's copy-and-patch), monitor execution to learn
whether the query is short or long, and switch to the heavy LLVM-vectorised
path only if the runtime predicts the switch will pay off.

**This is research-grade.** Defer until everything else lands and you have a
specific motivating workload that the static cost-gate (spec 04) misplans.

## Background

Umbra (Neumann et al., CIDR 2020) has a two-tier execution model:

1. **Tier 1:** Bytecode interpreter, no compilation. Starts immediately.
2. **Tier 2:** LLVM-compiled native code. Compiles in the background while
   tier 1 runs.

A query that finishes during tier 1's runtime never pays compile cost.
A query that runs long enough to amortize compile cost transparently switches
to tier 2 mid-execution.

Our analogue would be:
- **Tier 0:** PG's standard interpreter (already there, wired by the cost
  gate).
- **Tier 1:** Copy-and-patch JIT (spec 12 if landed).
- **Tier 2:** Cached LLVM-compiled native code (spec 03).

## Why deferred

The static cost-gate (spec 04) covers most of what adaptive execution buys.
The remaining wins are:
- Queries the planner badly mis-estimates (cost says "small" but actual is
  large). Adaptive switches mid-flight; static can't.
- Repeated queries where caching covers it but the *first* execution still
  pays cost. Adaptive lets first execution start without compile delay.

Both are real; neither is the dominant factor. Build only if data justifies it.

## What this spec produces (when activated)

A monitoring + switching framework:

1. **Per-query progress monitor.** During tier 0/1 execution, count tuples
   processed. Periodically (every N tuples or M ms) check whether predicted
   total work exceeds switching threshold.
2. **Background compilation.** When the monitor decides to switch, kick off
   tier 2 compilation in a background thread. **PostgreSQL is single-threaded
   per backend** — so this means a worker process or a deferred-work queue
   processed at safe yield points. This is the hard part.
3. **Switch point.** Once compilation finishes, the next "yield" in tier 0/1
   transfers control to tier 2 with the partial query state (already-emitted
   tuples don't repeat, hashtable state migrates).

State migration is the central hard problem. The interpreted-form hashtable
must transfer to the compiled-form hashtable, layouts being different.
Acceptable simplification for v1: only switch at clean operator boundaries
(after a complete scan, before the next pipeline stage).

## Pre-requisites before activating this spec

- Specific identified workload where static cost-gating mispredicts.
- Spec 03 (cache) and spec 09 (vector dialect) are landed and stable.
- A/B data showing >5% workload wall time wasted on misgated queries.

## Acceptance criteria (when activated)

- Threshold-monitor implementation runs <1% overhead during tier 0/1.
- A/B (mixed workload, must construct a representative one):
  - 90th-percentile query latency improves vs. static gate.
  - Median latency unchanged (no regression on already-correctly-gated queries).
- Correctness: every query produces identical results regardless of which
  tier(s) it runs in.

## Risks

- PG's process-per-backend model makes background compilation invasive.
  May require a worker-process protocol.
- State migration across operator boundaries is fragile. Bugs here mean
  silent wrong results.
- The complexity probably doesn't pay for itself unless the workload
  specifically fits.

## A/B test

Required at activation. Spec ID prefix: `13-adaptive`.

Construct a synthetic workload mixing queries the cost-gate handles correctly
with queries it misgates. Measure 50th and 90th percentile latency. Static
baseline vs adaptive.

## Rollback

`pgx_lower.adaptive_execution = off` falls back to static gating.
