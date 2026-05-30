## 🟡 MAYBE — suspected JIT-compile-noise outlier on **q01** (-54.7%); geomean -0.1% is healthy and no other query regresses past -10%. Re-run `just bench-merge` to confirm before treating as real.

- **Geomean speedup vs baseline:** -0.1% ⚪
- **Scale factor:** 0.01, iterations: 5
- **Baseline:** `unknown` (2026-04-19T01:55:38.889290)
- **Current:**  `unknown` (2026-04-19T02:38:04.562187)

![bench diff](https://raw.githubusercontent.com/zyros-dev/pgx-lower/harness-fixes-3/benchmarks/pr-9-harness-fixes-3.png)

PG column is PostgreSQL-only runtime from the current run — reference for absolute pgx-lower cost, not part of the A/B.

| Query | PG (ms) | pgx baseline (ms) | pgx current (ms) | Δ | ✓ |
|-------|--------:|------------------:|-----------------:|--:|:--|
| q01 | 58.5 | 366.7 | 567.3 | -54.7% 🔴 | ✅ |
| q02 | 6.7 | 696.1 | 579.2 | +16.8% 🟢 | ✅ |
| q03 | 10.8 | 508.8 | 502.3 | +1.3% 🟢 | ✅ |
| q04 | 4.0 | 227.7 | 228.7 | -0.4% ⚪ | ✅ |
| q05 | 39.5 | 376.6 | 380.7 | -1.1% 🔴 | ✅ |
| q06 | 8.5 | 127.4 | 129.6 | -1.8% 🔴 | ✅ |
| q07 | 7.0 | 409.6 | 404.0 | +1.4% 🟢 | ✅ |
| q08 | 71.8 | 487.4 | 490.5 | -0.6% ⚪ | ✅ |
| q09 | 26.4 | 474.6 | 477.0 | -0.5% ⚪ | ✅ |
| q10 | 15.9 | 411.2 | 387.2 | +5.8% 🟢 | ✅ |
| q11 | 3.0 | 327.3 | 318.0 | +2.9% 🟢 | ✅ |
| q12 | 13.1 | 236.2 | 234.3 | +0.8% ⚪ | ✅ |
| q13 | 8.1 | 189.8 | 192.9 | -1.6% 🔴 | ✅ |
| q14 | 8.1 | 180.4 | 182.7 | -1.3% 🔴 | ✅ |
| q15 | 8.8 | 264.6 | 260.2 | +1.7% 🟢 | ✅ |
| q16 | 4.9 | 292.1 | 288.8 | +1.1% 🟢 | ✅ |
| q18 | 25.5 | 522.5 | 502.4 | +3.8% 🟢 | ✅ |
| q19 | 14.9 | 306.9 | 290.4 | +5.4% 🟢 | ✅ |
| q21 | 12.0 | 600.5 | 574.0 | +4.4% 🟢 | ✅ |
| q22 | 3.3 | 273.1 | 271.0 | +0.8% ⚪ | ✅ |
