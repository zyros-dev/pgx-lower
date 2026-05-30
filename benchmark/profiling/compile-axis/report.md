# pgx-lower Compile-Axis Composition Report — SF=1.0

**Generated:** 2026-05-17  
**Source data:** `benchmark/profiling/phase-timing/sf1-report.md` (SF=1.0, 5 iterations, medians, 20 queries; q17/q20 skipped)  
**Skipped:** q17, q20

## CONTEXT: compile vs execution scale

Execution dominates compile at SF=1: the exec/compile ratio has a **median ≈ 69×** across the 20 queries (range ≈ 12×–206×). (For reference, the Phase-2 gate's 71× / 175× figures were exec/**lowering** for q01/q18 — a different ratio.) This report characterises only the *internal composition of that small compile cost* — it does not imply compile is a dominant or performance-critical path at SF=1.

---

## Per-Query Compile-Phase Breakdown

All times are median milliseconds over 5 iterations.  
`mlir_ms` = setup + translate + lowering.  
`total_compile` = setup + translate + lowering + jit.  
`mlir_frac` = mlir_ms / total_compile (MLIR-side fraction of compile).

| Query | setup ms | translate ms | lowering ms | jit ms | total_compile ms | mlir_ms | mlir_frac |
|-------|----------|--------------|-------------|--------|-----------------|---------|-----------|
| q01   |      1.7 |          3.0 |       328.7 |   38.9 |           372.3 |   333.4 |     0.896 |
| q02   |      1.8 |          4.2 |       386.2 |   63.0 |           455.2 |   392.2 |     0.862 |
| q03   |      1.5 |          1.6 |       131.9 |   24.8 |           159.8 |   135.0 |     0.845 |
| q04   |      1.6 |          1.6 |       133.2 |   24.4 |           160.8 |   136.4 |     0.848 |
| q05   |      1.7 |          4.3 |       232.5 |   38.6 |           277.1 |   238.5 |     0.861 |
| q06   |      1.7 |          1.8 |        56.0 |    9.5 |            69.0 |    59.5 |     0.862 |
| q07   |      1.7 |          5.6 |       266.0 |   43.0 |           316.3 |   273.3 |     0.864 |
| q08   |      1.8 |          8.8 |       327.2 |   48.6 |           386.4 |   337.8 |     0.874 |
| q09   |      1.8 |          5.3 |       256.5 |   41.9 |           305.5 |   263.6 |     0.863 |
| q10   |      1.6 |          3.5 |       253.0 |   45.1 |           303.2 |   258.1 |     0.851 |
| q11   |      1.5 |          4.9 |       237.0 |   36.7 |           280.1 |   243.4 |     0.869 |
| q12   |      1.6 |          3.8 |       152.7 |   23.1 |           181.2 |   158.1 |     0.873 |
| q13   |      1.8 |          1.4 |       129.5 |   22.4 |           155.1 |   132.7 |     0.856 |
| q14   |      1.6 |          3.7 |       100.6 |   15.5 |           121.4 |   105.9 |     0.872 |
| q15   |      1.7 |          2.4 |       166.0 |   25.0 |           195.1 |   170.1 |     0.872 |
| q16   |      2.0 |          1.7 |       153.2 |   27.8 |           184.7 |   156.9 |     0.850 |
| q18   |      1.7 |          2.5 |       234.3 |   38.5 |           277.0 |   238.5 |     0.861 |
| q19   |      1.7 |          6.1 |       145.4 |   24.5 |           177.7 |   153.2 |     0.862 |
| q21   |      1.6 |          3.9 |       298.7 |   47.4 |           351.6 |   304.2 |     0.865 |
| q22   |      1.5 |          2.3 |       212.9 |   34.2 |           250.9 |   216.7 |     0.864 |

---

## Aggregate: Geomean MLIR-Side Fraction

**Geomean (setup+translate+lowering) / (total compile) = 0.8633 (86.3%)**

Computed as the geometric mean across all 20 queries of their individual `mlir_frac` values.

Individual fractions range from **0.845** (q03) to **0.896** (q01) — tight band, consistent across all query complexities.

---

## Decision Gate

**Gate criterion:** geomean MLIR-side fraction >= 0.80

**Result: YES — 0.863 >= 0.80**

**Verdict:** Compile-axis composition is MLIR-side dominated. The LLVM/JIT phase (`jit_ms`) accounts for ~13.7% of total compile time; the three MLIR phases (setup + translate + lowering) account for ~86.3%. Within the MLIR portion, lowering dominates: setup is ~1–2 ms, translate is ~2–9 ms, lowering is ~56–386 ms.

---

## Sanity Checks

- **setup** is uniformly small: 1.5–2.0 ms across all queries.
- **lowering** is the dominant compile component: 56–386 ms (matches prior single-query instrumentation of ~230–330 ms).
- **jit** is the minor LLVM phase: 9.5–63 ms (~14% of compile on average).
- q02 has the highest jit_ms (63 ms) and q06 has the smallest total compile (69 ms, simplest query). Both remain within the expected pattern.

---

## Summary

| Metric | Value |
|--------|-------|
| Geomean MLIR fraction | **86.3%** |
| Gate (>= 80%) | **PASS** |
| lowering share of compile | ~84% (dominant) |
| jit share of compile | ~14% (minor) |
| setup share of compile | ~1% (negligible) |
| translate share of compile | ~2% (negligible) |
| exec/compile ratio (context) | 12×–206×, median ≈69× |

The H1 hypothesis — "MLIR-side phases >> LLVM/JIT phase" — is validated at SF=1. MLIR lowering is the primary compile-time cost; LLVM/JIT is secondary. This characterises the *composition* of a cost (compile) that is itself minor relative to execution.
