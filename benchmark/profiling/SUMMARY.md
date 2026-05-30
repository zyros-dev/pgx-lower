# pgx-lower Slowness Validation — SUMMARY (gate doc, 2026-05-17)

All findings spec-reviewed. Single-query scope: **q01 @ SF=1** (NUMERIC-heavy aggregate; the canonical TPC-H compile-engine stress query). Generalises to NUMERIC-heavy aggregates; non-NUMERIC / join-heavy / string-date queries are **uncharacterised** — see Caveats.

## Axis 1 — Compile cost
- Hypothesis: compile (~236 ms MLIR) is a dominant cost; SF too small to see execution.
- Verdict: **execution ≫ compile at realistic scale.** SF=1: exec/compile median ≈ **69×** (range ~12–206×). Compile composition is MLIR-side ≈ **86%** vs LLVM/jit ≈14% (geomean, gate ≥0.8 PASS) — so *of* the small compile cost, MLIR lowering dominates, but compile is ~a rounding error vs execution at SF≥1.
- Fix it justifies: plan-shape compile cache (spec 03) is a **small-query / benchmark-warm** fix only — NOT the lever to beat PG.

## Axis 2 — Execution (the crux)
- Hypothesis (the whole project's framing): opaque per-tuple `extract_field` FFI wall reintroduces materialisation/dispatch.
- Verdict: **REFUTED** — decisively, RelWithDebInfo + jitdump-symbolised profile, spec-reviewed including the inlining steelman. The JITed query loop ("main") is only **4.28%** of execution; `extract_field`/`get_*_field_mlir` are ABSENT in Debug *and* optimised; ≤~2.5% upper bound for any inlined decode. **The generated code is not the problem.**
- Verified real bottleneck taxonomy (q01@SF=1, RelWithDebInfo):
  1. **`PGX_IO` instrumentation own-goal ≈ 19.6%** — `should_log()` + `std::optional<ScopeLogger>` ctor *per tuple-op*. `PGX_RELEASE_MODE` does **NOT** gate `PGX_IO` (only `PGX_HOT_LOG`); `PGX_IO` in `include/pgx-lower/utility/logging.h:228-231` has no guard (verified in source). **#1 FIX:** gate `PGX_IO` on `#ifndef PGX_RELEASE_MODE` — ~1 line, ~19.6% of cycles, zero correctness impact (logging already GUC-off).
  2. PG tuple decode: `heap_deform_tuple` 11.1% + `AllocSetAlloc` 4.5% ≈ **16%** (structural; spec 05 decode-at-scan partial).
  3. NUMERIC→`__int128`: `numeric_to_i128` 9.2% + `__divti3` 2.4% ≈ **11.6%** (LingoDB-inherited data-type conversion, `NumericConversion.cpp:99-146`; spec 06).
  4. Per-column batch decode: `DataSourceIteration::access` 8.0% + `process_tuple_into_batch` 5.2% ≈ **13%**.
- Optimisation context: query 23 s→4 s, IPC 1.60→2.22, stalls 36%→19% (Debug was NOT production-representative).

## Axis 3 — Data-type misalignment
- 7 LingoDB-vs-PG mismatches enumerated from source (see `types/report.md`). Only **M1 (NUMERIC→i128) is a measured perf cost (~11.6%)**. M2–M7 (null-flag inversion, BPCHAR padding, INTERVAL-month `ereport(ERROR)`, DATE/TIMESTAMP unit double-convert, per-cell string palloc) are **correctness/robustness debt**, not measured here (q01 has no hot string/date path) — they matter for other query classes.

## Recommended fix ordering (revised by evidence)
1. **Gate `PGX_IO` on `PGX_RELEASE_MODE`** — biggest, cheapest, zero-risk (~19.6%).
2. NUMERIC→i128 (M1) — scale-adjust loop + detoast redundancy (~11.6%); spec 06.
3. PG decode (heap_deform_tuple/alloc) — spec 05 decode-at-scan.
4. Correctness debt before relevant workloads: M4 INTERVAL-month (silent corruption / hard error), M3 BPCHAR padding (CHAR equality filters, e.g. TPC-H `l_returnflag`).
5. Compile cache (spec 03) / SubOp / FFI work: **de-prioritised** — FFI-wall refuted; compile is not the realistic-scale lever.

## Method integrity (the meta-result)
Three self-contamination traps were caught by review before they propagated: (1) Task-1 phase timers missed per-query MLIRContext/setup; (2)+(3) execution profiles polluted by logging — `postgresql.auto.conf` carried `pgx_lower.log_enable='on'` leaked via a phase-timing `ALTER SYSTEM SET` (now reset-guarded in the harness). Each "obvious" early conclusion (compile cheap; should_log is the bug; FFI is the bug) was an artifact of an unsound measurement. The FFI-wall — taught, wikied, and used to architect this very plan — did not survive its own investigation. That is the central lesson ("measure; the assumed dominant term is rarely the real one") proven end-to-end.

## Caveats / what is NOT established
- Single query (q01) at SF=1. Recommend confirming on a join-heavy query (q18) and a string/date query before acting on anything beyond fix #1 (PGX_IO, which is query-independent).
- "5–106× slower than PG" is **stale** (old SF=0.01/0.5 benches). SF=1 phase-timing suggested pgx ≈ PG within a few % for many queries — *possible near-parity*, not rigorously established.
- Profiling tooling (perf+jitdump) requires `kernel.perf_event_paranoid≤1` on thor (set + persisted) and the in-container perf binary; see `tooling.md`.
