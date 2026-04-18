# Spec 12 — Copy-and-patch backend (deferred / research)

**Tier:** 4 (research)
**Stack on:** 01–04 landed
**Blocks:** —
**Estimated effort:** 4–6 weeks (research-grade)

## Goal

Add a copy-and-patch JIT backend alongside the LLVM ORC backend, so cold
small queries that don't qualify for the cost gate (spec 04) can still get
JITed code in *microseconds* instead of milliseconds. CPython 3.13 uses this
technique; the academic paper is "Copy-and-Patch Compilation" (Xu & Kjolstad,
OOPSLA 2021).

**This is a research spec.** Don't start it until specs 01–04 are settled and
you have data showing cold-small queries are still a meaningful fraction of
the workload. The simpler answer (cost-gate to PG) likely covers most of it.

## Background

Copy-and-patch compiles by:

1. **Build time:** Use Clang to compile every "stencil" (a primitive operation
   like "scan i64 column", "filter >", "sum aggregate") to position-independent
   machine code, with placeholder slots for runtime constants.
2. **JIT time:** `memcpy` the stencils into a code buffer, then patch in the
   actual constants. No optimization passes run at JIT time. Compilation
   time is essentially zero — microseconds, not milliseconds.

Trade-off: generated code is ~80% of LLVM's quality (no cross-stencil
inlining, no global optimization). For *cold* small queries, this is the right
trade-off because LLVM's compile cost would exceed PG's interpreted execution
time.

## Why deferred

After specs 01–04:
- Spec 03 (cache) covers all warm queries.
- Spec 04 (cost-gate) routes cold-small queries to PG's interpreter.

What's left? Cold queries that are too expensive for PG's interpreter but
too cheap to amortize LLVM compile cost. This is a narrow band, and the
right metric to measure is "how much wall time across our workload sits in
this band." If it's <5%, copy-and-patch isn't worth the engineering cost.

## What this spec produces (when activated)

A first-cut implementation covering 4–5 stencils:
- Sequential scan (per int/decimal column type).
- Filter with comparison.
- Sum aggregate.
- GROUP BY hash insertion.
- Result projection.

Architecture:

1. **Stencil definition.** A `.cpp` file per stencil, hand-tuned to compile
   to a small, position-independent code blob with named placeholder symbols.
2. **Build-system integration.** CMake invokes Clang with specific flags
   (`-fno-pic`, custom linker scripts) and dumps the compiled stencil's
   machine code + relocation table to a header.
3. **Runtime patcher.** Reads the compiled stencils into a JIT code buffer,
   patches placeholders with runtime values (column offsets, constant comparison
   values, function pointers), flushes I-cache.
4. **Selection logic.** Extends spec 04's gate: if total_cost < THRESHOLD_LLVM
   AND total_cost > THRESHOLD_PG, use copy-and-patch. Otherwise existing rules.

## Pre-requisites before activating this spec

- Spec 04's cost-gate calibration data is in. Specifically: the histogram of
  query total_cost from a representative workload, showing the band of
  queries that hit "neither cached nor cost-gated."
- That band represents >5% of total workload wall time.
- Otherwise this spec stays deferred.

## Acceptance criteria (when activated)

- Stencils for the 5 listed primitives exist and pass correctness tests.
- A/B (cold, SF=0.01):
  - **q06 cold time: ≥3× faster** than current cold q06 (which already
    benefits from spec 04 if cost-gated).
  - q01, q03, q12: marginal or no impact (these miss the cost-gate-out
    threshold and use the LLVM path).
- Compile-time per query: <1ms for the 5-stencil set. (LLVM is 100–250ms.)

## Risks

- Stencils are platform-specific (x86_64 vs ARM64). Initial implementation
  x86_64-only.
- Maintenance burden: every new operation we want to support cold needs a
  new stencil. Don't build this unless it's clearly worth it.
- Code quality is lower than LLVM. Running a query enough times that it
  enters the cache (spec 03) means the worse-quality stencil code runs many
  times before being replaced. Either: invalidate copy-and-patch entries
  aggressively, or accept the slight per-query cost.

## A/B test

Required only at activation time. Spec ID prefix: `12-copyandpatch`.

Specific runs:
- Cold-only sweep (drop caches between queries) for the cost-gate-band queries.
- Compile-time microbench: time `memcpy + patch` vs `LLVM compile`.

## Rollback

Disabled by default behind `pgx_lower.enable_copy_and_patch` GUC. Spec 04's
gating still applies in front. Trivial to revert.
