# Spec 02 — Enable LLVM auto-vectorization

**Tier:** 1 (free wins)
**Stack on:** `main` (independent of 01)
**Blocks:** informs scope of 09 (manual vector dialect work)
**Estimated effort:** 1 day (most of it is benchmarking)

## Goal

LLVM's loop vectorizer, SLP vectorizer, and loop unroller are explicitly
disabled in our JIT optimization pipeline. Turn them on and measure. This is
the cheapest possible test of "how much does our compiled code already benefit
from SIMD?" before committing weeks to a manual MLIR `vector` dialect lowering
(spec 09).

## Background

`src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp:305-308`:

```cpp
llvm::PipelineTuningOptions PTO;
PTO.LoopUnrolling      = false;
PTO.LoopVectorization  = false;
PTO.SLPVectorization   = false;
```

These are passed to `llvm::PassBuilder PB(TM, PTO)` at line 310. The remainder
of the optimization pipeline (lines 326–337) is a manually-assembled function
pass list (SROA, InstCombine, Promote, LICM, Reassociate, GVN, SimplifyCFG)
that doesn't include the vectorization passes — so flipping `PTO` alone won't
help unless we also use the standard pipeline.

## What to change

### 1. Flip the flags

```cpp
PTO.LoopUnrolling      = true;
PTO.LoopVectorization  = true;
PTO.SLPVectorization   = true;
```

### 2. Switch to PassBuilder's standard pipeline

The current manually-assembled `FunctionPassManager` (lines 324–337) is a
hand-rolled subset of `-O2`. To get vectorization to actually fire, we need to
use the standard pipeline:

```cpp
llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(
    llvm::OptimizationLevel::O2);
MPM.run(*module, MAM);
```

Replace lines 323–349 (the entire `FunctionPassManager` construction and
per-function loop) with the module-level pipeline above. Keep the
`install_fatal_error_handler` block (lines 278–284) and the
`TargetMachine` setup (lines 286–303).

### 3. Verify vectorization actually fires

Add a one-time IR dump before and after the pipeline (gated on
`pgx_lower.log_ir` GUC) to confirm vector instructions appear. Look for
`<4 x i64>`, `<8 x i32>`, etc. in the post-optimization IR for a hot scan loop.

The existing `dumpLLVMIR(module, "LLVM IR AFTER OPTIMIZATION PASSES", ...)`
call at line 352 already does post-optimization dump — just verify it's reached
under the new pipeline.

### 4. Compile-cost ceiling

Auto-vectorization adds compile time. On q01 we expect ~30–80ms additional
LLVM time. **This is acceptable for warm queries (spec 03 caches it away)
but visible on cold queries.** Document this trade-off in the PR. Spec 04
(cost-gating) is the long-term answer for cold-small queries.

## Acceptance criteria

- Build clean on thor.
- All existing tests pass.
- Inspect IR: at least one TPC-H query (q01 or q06) shows vector instructions
  in the post-optimization dump.
- A/B (warm, SF=0.01, 4-query subset):
  - **q01: ≥10% faster** (heavy aggregate, scan-bound — strong vectorization target)
  - **q06: ≥10% faster** (simplest scan+sum, easiest target)
  - **q03, q12: any direction acceptable; report the number**
- A/B (cold, same): cold-query regression up to +50ms is acceptable; > +100ms
  on q06 is a red flag, dig in.
- Branch-prediction profile run: report `branches`, `branch-misses`,
  `LLC-loads`, `LLC-load-misses`, `ipc` from `perf_stats` table for q01 and
  q06. The thesis baseline says branch misprediction was already at 0.16% —
  vectorization may push that higher (more conditional folds → fewer branches
  total, but each is harder to predict). Capture the data either way.

## Risks

- LLVM's vectorizer can produce code that crashes on certain malformed IR
  patterns. Run the full unit test suite, not just TPC-H.
- The current pipeline doesn't include `mem2reg`-equivalent before
  vectorization — `O2` includes it but check ordering. If you see no vector
  instructions in IR, that's likely the cause.
- Some queries may *regress* if our scan loops aren't auto-vectorizable
  (per-tuple FFI calls into `get_int64_field_mlir` are likely opaque to the
  vectorizer). That's data, not failure — it's the input to spec 09's design.

## A/B test

See `specs/ab-test-template.md`. Spec ID prefix: `02-vec`.

**Required runs:**
1. Standard 4-query sweep (cold + warm).
2. Branch-prediction profile (perf counters).

**Comparison report in PR:** include the perf-stat table comparing pgx_enabled=0
(unchanged baseline) vs pgx_enabled=1 (this branch) for IPC, branch-miss rate,
LLC-miss rate.

## Decision flag for spec 09

In the PR description, answer: **"Did LLVM auto-vec deliver enough wins that
the manual vector dialect work in spec 09 should be deferred or descoped?"**
This is the gate spec 09 hangs on.

## Rollback

Three lines in `jit_execution_engine.cpp` and a pipeline swap. Trivial revert.
