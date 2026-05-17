# Validate pgx-lower Slowness Theories — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empirically prove (or disprove) *why* pgx-lower runs 5–106× slower than stock PostgreSQL, by isolating compile-time from execution-time at a scale where execution matters, then profiling each, before changing any engine code.

**Architecture:** Two cost axes, validated independently. (A) Compile axis — already strongly indicated as ~236 ms MLIR lowering vs ~15 ms LLVM; confirm with phase timing. (B) Execution axis — *theorised but unvalidated*: opaque per-tuple `extract_field` FFI reintroduces materialization + an optimizer wall. The blocker for (B) is that at SF=0.01 the compile cost drowns execution; the plan fixes measurement *first*, then profiles execution-only with `perf`. A third strand investigates LingoDB-inherited data-type misalignment. Every fix is gated on a validated bottleneck.

**Tech Stack:** pgx-lower (C++/MLIR/LLVM 20), PostgreSQL 17.6, TPC-H, Docker-on-thor build (mac is edit-only; mutagen sync; `just` recipes queue via `tsp` on thor; thor SSH alias `comfy`). Profilers: `perf` (Linux, AMD-compatible — thor is AMD). **Magic Trace is unavailable: it requires Intel Processor Trace; thor is AMD.**

---

## Machines — where every step runs (NO heavy work on the Mac, EVER)

| Machine | Role | What runs here |
|---|---|---|
| **Mac** (this host) | **Edit only.** mutagen *alpha*. | File edits, reading code, writing plan/report `.md`, git ops on the worktree. **NEVER:** compile, PostgreSQL, the extension, benchmarks, `perf`, anything CPU/IO-heavy. |
| **thor** (`comfy`, user `zel`) | **All heavy work.** mutagen *beta*; Docker image with LLVM 20 / MLIR 20 / PG 17.6. | Every `just` recipe (queued via `tsp`): `compile`, `bench`, `utest`, `profile-exec`; PostgreSQL; the JITed extension; **all `perf` profiling runs inside the thor container**. |
| **Odin** | **NOT used.** | Considered earlier only as a Magic Trace host. Magic Trace is rejected (needs Intel PT). So Odin is unnecessary and too small to host the toolchain anyway. |

**Mechanics:** edit on the Mac → mutagen syncs to thor (allow ~1s after an edit) → `just <recipe>` queues + runs on **thor** via `tsp`. The worktree gets its own mutagen session (`pgx-lower-validate-slowness`) from `just worktree-new`. Artifacts are written on thor and sync back to the Mac for review/commit.

**Rule for reading this plan:** any step that says `Run: just ...` (or `perf ...`) executes **on thor**, never the Mac. Steps that only edit/create files happen on the Mac and sync over. No build, run, benchmark, or profile ever touches the Mac.

---

## Constraints & ground rules (read before any task)

- **Everything builds/runs on thor; the Mac is edit-only.** See the Machines table above — this is non-negotiable. Use `just` recipes (`just --list` for the surface); they queue on thor via `tsp`.
- **No git history rewrite.** Work from a fresh branch off current `main` (audit decided: the agentic-era engine output is tiny and partly correct — spec 05 — nothing to revert).
- **Validation before fixes.** Tasks in Phases 0–5 change *no engine code*. The fix roadmap (Phase 6) is a gated outline, not yet actionable steps — it gets its own plan once Phase 5 says which bottleneck is real.
- **Every experiment commits its artifact** (timing JSON, `perf` data, flamegraph SVG, report MD) so results are reproducible and reviewable.
- **Decision gates are hard.** If a hypothesis fails its criterion, stop and record it — do not proceed to the fix that assumed it.

---

## File Structure

- `docs/superpowers/plans/2026-05-17-validate-slowness-theories.md` — this plan.
- `benchmark/profiling/` — **new** dir: all captured artifacts (timings, perf data, flamegraphs, reports). One subdir per experiment.
- `justfile` — add `profile-exec` and `phase-timing` recipes (Phase 1/2).
- `src/pgx-lower/execution/mlir_runner.cpp` — instrument the compile/execute boundary (Phase 1, timing only — no behaviour change).
- `src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp` — the `compile()` vs `execute()` calls; the phase boundary lives here / in `mlir_runner.cpp`.
- `src/pgx-lower/runtime/tuple_access.cpp` — `extract_field<T>` / `get_*_field_mlir` (the per-tuple FFI under suspicion; read-only in validation phases).
- `benchmark/` — existing TPC-H harness; Phase 1 adds an SF≥1 path and compile/exec split reporting.

---

## Phase 0 — Workspace & green build

### Task 0: Branch, worktree, sanity build

**Files:** none modified (workspace only).

- [ ] **Step 1: Create an isolated worktree off current main**

Run: `just worktree-new validate-slowness`
Expected: a new worktree + its own mutagen session (`pgx-lower-validate-slowness`), branch `validate-slowness` off `main`.

- [ ] **Step 2: Confirm the branch point is current main, not a reset**

Run: `git -C <worktree> log --oneline -1 && git -C <worktree> merge-base HEAD origin/main`
Expected: HEAD is a fresh branch whose merge-base is current `main` tip (`e246028` or later). No history rewritten.

- [ ] **Step 3: Clean build on thor**

Run: `just compile`
Expected: build succeeds (queued via tsp on thor). Capture the build log.

- [ ] **Step 4: Smoke-run one TPC-H query end-to-end through pgx-lower**

Run: `just bench` (default config) and confirm at least Q01 produces a result whose hash matches PG (existing correctness gate).
Expected: PASS (correctness), regardless of speed. This proves the pipeline runs before we measure it.

- [ ] **Step 5: Commit the workspace marker**

```bash
mkdir -p benchmark/profiling
echo "validation workspace — see docs/superpowers/plans/2026-05-17-validate-slowness-theories.md" > benchmark/profiling/README.md
git add benchmark/profiling/README.md
git commit -m "chore: profiling workspace for slowness validation"
```

---

## Phase 1 — Fix measurement (the prerequisite)

> Rationale: at SF=0.01 the ~236 ms compile drowns execution, so the execution-axis theory is *unobservable*. We must (a) split compile vs execution timing, (b) run at SF≥1.

### Task 1: Instrument the compile/execute phase boundary (timing only)

**Files:**
- Modify: `src/pgx-lower/execution/mlir_runner.cpp` (around the lowering-pipeline call and the JIT compile/execute calls — locate with Step 1).
- Modify: `src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp` (the `compile()` and `execute()` entry points).

- [ ] **Step 1: Locate the exact boundary**

Run: `grep -n "runCompleteLoweringPipeline\|executeJITWithDestReceiver\|->compile(\|->execute(" src/pgx-lower/execution/mlir_runner.cpp src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp`
Expected: the call sites that delimit ① translate, ② lowering staircase, ③ LLVM/JIT compile, ④ native execution.

- [ ] **Step 2: Add monotonic-clock timing around the four sub-phases**

Wrap each region with `std::chrono::steady_clock` start/stop; accumulate into a struct and emit one structured log line per query:
`PGXL_PHASE_TIMING setup_ms=.. translate_ms=.. lowering_ms=.. jit_ms=.. exec_ms=.. query=<id>`
Five phases: **setup** (MLIRContext ctor + 14 dialect loads), **translate** (PG AST → RelAlg), **lowering** (RelAlg → LLVM dialect), **jit** (JITEngine ctor + register_dialects + ORC compile), **exec** (native run). Total compile = setup+translate+lowering+jit.
Guard behind the existing logging GUC so it is opt-in and adds no hot-path cost when off. **No behaviour change** — timing only.

- [ ] **Step 3: Build**

Run: `just compile`
Expected: success.

- [ ] **Step 4: Verify the log line appears and the split is sane**

Run: `just bench` with the timing GUC on, one query.
Expected: a `PGXL_PHASE_TIMING` line with all five fields; sanity check: `(setup+translate+lowering)` vs `jit_ms` indicates which side dominates compile (prior indication: MLIR-side ≫ jit).

- [ ] **Step 5: Commit**

```bash
git add src/pgx-lower/execution/mlir_runner.cpp src/pgx-lower/execution/jit_engine/jit_execution_engine.cpp
git commit -m "feat: opt-in per-phase timing (translate/lowering/jit/exec)"
```

### Task 2: SF≥1 benchmark path with compile/exec split report

**Files:**
- Modify: `benchmark/` harness scripts (the TPC-H runner + report generator).
- Create: `benchmark/profiling/phase-timing/` (artifact dir).

- [ ] **Step 1: Add an SF=1 (and SF=10 optional) data-generation + run path**

Extend the harness so scale factor is a parameter and SF=1 is runnable; keep idempotent TPC-H caching. Document the exact invocation.

- [ ] **Step 2: Add a report that tabulates, per query: PG total, pgx translate/lowering/jit/exec, pgx total**

The report MUST present compile and execution **separately**, never lumped. Output to `benchmark/profiling/phase-timing/sf1-report.md`.

- [ ] **Step 3: Run SF=1, ≥5 iterations, capture**

Run the SF=1 benchmark.
Expected artifact: `sf1-report.md` with the per-query split.

- [ ] **Step 4: Decision gate — is execution now observable?**

Criterion: at SF=1, for the heavier queries (e.g. Q01, Q18), `exec_ms` is at least the same order of magnitude as `lowering_ms` (execution no longer dwarfed by compile).
- If YES → execution-axis is measurable; proceed to Phase 2.
- If NO (compile still dominates even at SF=1) → escalate scale to SF=10 for the heavy queries before proceeding; record this.

- [ ] **Step 5: Commit the artifact**

```bash
git add benchmark/ benchmark/profiling/phase-timing/sf1-report.md
git commit -m "feat: SF>=1 benchmark with separated compile/exec reporting + first split report"
```

---

## Phase 2 — Profiler selection & wiring

### Task 3: Confirm tooling and wire a profiling recipe

**Files:**
- Modify: `justfile` (add `profile-exec`).
- Create: `benchmark/profiling/tooling.md` (decision record).

- [ ] **Step 1: Record the tooling decision (research output)**

Write `benchmark/profiling/tooling.md` stating: Magic Trace **rejected** (needs Intel PT; thor is AMD); **`perf` chosen** (AMD-compatible, sampled call-graph + hardware counters — the same counter family the thesis used); AMD uProf noted as fallback for deeper µarch counters. Include the exact `perf` version present in the thor container (`perf --version`).

- [ ] **Step 2: Confirm JIT symbolization works**

The hot code is JITed into the PG process. Verify `perf` can attribute samples to JITed frames and to `extract_field` (the extension exports symbols via `-Wl,--export-dynamic`). If JIT frames are unsymbolized, enable `perf` JIT support (`perf inject --jit` / `jitdump`) or, minimally, confirm the *FFI* symbol `extract_field` resolves (it is a real exported symbol, so it will). Record findings in `tooling.md`.

- [ ] **Step 3: Add `just profile-exec` recipe**

Recipe runs, inside the thor container, a single chosen query at SF≥1 under:
`perf record -g --call-graph dwarf -o <artifact>.data -- <pg run of the query>`
and
`perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,stalled-cycles-frontend,stalled-cycles-backend -- <same>`
Output artifacts into `benchmark/profiling/<experiment>/`.

- [ ] **Step 4: Dry-run the recipe on one query**

Run: `just profile-exec QUERY=q01 SF=1`
Expected: a `.data` file + a `perf stat` text capture, non-empty, symbols present for `extract_field`.

- [ ] **Step 5: Commit**

```bash
git add justfile benchmark/profiling/tooling.md
git commit -m "feat: perf profiling recipe + tooling decision record"
```

---

## Phase 3 — Validate the COMPILE axis

### Task 4: Confirm 236-vs-15 (MLIR lowering dominates compile)

**Files:** Create `benchmark/profiling/compile-axis/report.md`.

- [ ] **Step 1: Hypothesis**

H1: of total compile time, the MLIR-side cost (setup+translate+lowering) ≫ LLVM/JIT codegen (`jit_ms`) — prior indication ~236 ms vs ~15 ms. `setup_ms` (MLIRContext ctor + 14 dialect loads) is now separately visible and suspected to be a large fraction of that MLIR-side total.

- [ ] **Step 2: Measure across all TPC-H queries at SF=1**

Use the Phase-1 phase timing; aggregate `(setup_ms + translate_ms + lowering_ms)` vs `jit_ms` (geomean + per-query) into `compile-axis/report.md`. Also break out `setup_ms` alone so the dialect-load cost is quantified.

- [ ] **Step 3: Decision gate**

Criterion: `(setup_ms + translate_ms + lowering_ms) / (setup_ms + translate_ms + lowering_ms + jit_ms)` ≥ 0.8 geomean — i.e. MLIR-side cost (setup+translate+lowering) ≫ jit.
- YES → compile-axis bottleneck = MLIR-side (setup/translate/lowering), **validated**. The relevant fix is the plan/compile cache + lowering shave + context amortization (Phase 6), not a faster backend.
- NO → record the actual split; revise the compile-axis fix story accordingly.

- [ ] **Step 4: Commit**

```bash
git add benchmark/profiling/compile-axis/report.md
git commit -m "data: compile-axis validation (lowering vs jit split, SF=1)"
```

---

## Phase 4 — Validate the EXECUTION axis (the core unknown)

### Task 5: Profile execution-only; test the FFI-wall hypothesis

**Files:** Create `benchmark/profiling/exec-axis/`.

- [ ] **Step 1: Hypothesis**

H2: in the *execution* phase, per-tuple opaque `extract_field` (call overhead + register spill + the optimizer wall it creates) dominates the hot loop — i.e. execution is call/spill-bound, not compute-bound.

- [ ] **Step 2: Capture an execution-only profile**

Run: `just profile-exec QUERY=q01 SF=1` (and a scan-heavy query, e.g. q06) — gated to the post-JIT region (the phase boundary from Task 1; warm the compile if needed so the sample window is execution).
Expected artifact: `exec-axis/q01.data`, `exec-axis/q06.data`.

- [ ] **Step 3: Generate flamegraphs + top-symbols**

`perf report --stdio` and a flamegraph SVG into `exec-axis/`.
Expected: a ranked symbol list.

- [ ] **Step 4: Decision gate (the key result of the whole plan)**

Criterion for H2 supported: `extract_field` / `get_*_field_mlir` / their callees (heap access + the spill prologue/epilogue around them) are the top execution-time consumers (cumulatively the plurality of execution samples), **and** `perf stat` shows low IPC / high stalled-cycles in the loop (call/spill-bound, not ALU-bound).
- SUPPORTED → execution-axis bottleneck = per-tuple FFI, **validated**. Phase 6 fix order is justified.
- NOT SUPPORTED → record what *actually* dominates execution (e.g. numeric/decimal conversion, hash, memory). The fix roadmap must be re-derived from this; do **not** proceed to "remove FFI" on faith.

- [ ] **Step 5: A/B confirmation (decode inlined vs opaque)**

Build a *minimal* variant where the decode for one fixed-width column type is inlined (proof-of-concept only, not the real fix), re-profile the same query. Criterion: the `extract_field` cluster shrinks and execution `exec_ms` drops measurably for that query. This converts correlation into a causal A/B.

- [ ] **Step 6: Commit**

```bash
git add benchmark/profiling/exec-axis/
git commit -m "data: execution-axis validation (perf profile + decode-inline A/B)"
```

---

## Phase 5 — Investigate data-type misalignment

> The user-flagged concern: pgx-lower's type definitions were inherited from LingoDB and are misaligned with PostgreSQL's (e.g. inverted null-flag convention, decimal/i128 capping, BPCHAR→string length loss, INTERVAL hardcoded). This may add execution cost (per-tuple conversions) *and* be a correctness/robustness risk — relevant to sequencing fixes.

### Task 6: Quantify the type-misalignment tax and risk

**Files:** Create `benchmark/profiling/types/report.md`.

- [ ] **Step 1: Enumerate the known mismatches**

From `src/pgx-lower/runtime/` (NumericConversion, DateRuntime, StringRuntime) and `translation_core.cpp`, list each LingoDB-vs-PG type mismatch and where a per-tuple conversion is paid (date↔nanos, numeric↔i128, varlena/BPCHAR, null-flag inversion). Cite file:function.

- [ ] **Step 2: Attribute execution cost to conversions**

From the Phase-4 `perf` data, measure what fraction of execution samples sit in the conversion functions (separate from `extract_field` itself).

- [ ] **Step 3: Decision gate**

Classify: is type-conversion cost (a) a significant *execution* contributor (then it joins the execution-axis fix list), (b) mainly a *correctness/robustness* debt (then it gates the fix ordering but isn't a perf lever), or (c) negligible. Record the classification with the supporting numbers.

- [ ] **Step 4: Commit**

```bash
git add benchmark/profiling/types/report.md
git commit -m "data: data-type misalignment cost + risk classification"
```

---

## Phase 6 — Gated fix roadmap (outline only — NOT yet actionable)

> Do **not** start any of these until Phases 3–5 produce validated bottlenecks. Each becomes its own plan, scoped to what the data justified. Standing hypothesis to be tested, not assumed: doing these moves pgx-lower from 5–106× slower to competitive with PG on warm analytical TPC-H at SF≥1.

- [ ] **Decision synthesis:** write `benchmark/profiling/SUMMARY.md` — for each axis: hypothesis, validated? (Y/N + numbers), the fix it justifies. This is the gate document.

Gated fixes, in the order the analysis implies (revise per data):

1. **Data-type reconciliation first** *(if Phase 5 says it's a correctness/robustness gate or a real exec cost)* — align type defs with PG before building on them; everything downstream depends on correct, cheap decode. Likely prerequisite for clean FFI removal.
2. **Kill per-tuple opaque FFI** *(if H2 validated)* — bitcode-inline PG's own decode C / complete decode-at-scan (spec 05/08 lineage). The single execution-axis lever.
3. **Bring in the SubOp layer** — batching/vectorization (spec 11 rebase) so generated code is tight SIMD-friendly loops, not per-tuple glue. Large surgery; only after 1–2.
4. **Prune redundant MLIR lowering passes — carefully.** Empirically these are *entangled/load-bearing* (blunt removal broke queries because cost isn't threaded through); treat as untangling, not deletion.
5. **Plan-shape compile cache** *(compile-axis fix)* — skip the validated ~236 ms on repeats; also unlocks honest warm benchmarking.

Note: Umbra's *playbook* (Flying Start) does **not** port — it kills LLVM latency, but the validated compile cost is MLIR lowering, a different layer. Its *principles* (cheap first compile, measure-don't-predict, own the data layout) do.

---

## Self-Review

- **Spec coverage:** validate compile axis (Phase 3), validate execution/FFI axis (Phases 1–2 prerequisite + Phase 4), tool research incl. Magic-Trace-rejection + perf (Phase 2), scale-factor runs (Phase 1), data-types-first concern (Phase 5), branch-not-reset (Task 0), fix roadmap gated on validation (Phase 6) — all present and traced to the conversation's requirements.
- **Placeholder scan:** no "TBD"/"handle edge cases"; commands and decision criteria are concrete. Where exact line numbers would drift, a `grep`/locate step is the action (intentional, not a placeholder).
- **Consistency:** artifact paths under `benchmark/profiling/<experiment>/` throughout; phase-timing field names (`setup_ms/translate_ms/lowering_ms/jit_ms/exec_ms`) used consistently across Tasks 1, 3, 4; the compile/exec-separation rule is invariant from Phase 1 onward; total compile = setup+translate+lowering+jit.
