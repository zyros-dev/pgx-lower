## 🟡 MAYBE — in the noise band

- **Geomean speedup vs baseline:** -0.5% ⚪
- **Scale factor:** 0.01, iterations: 5
- **Baseline:** `unknown` (2026-04-19T00:57:37.747930)
- **Current:**  `unknown` (2026-04-19T01:13:41.392817)

![bench diff](https://raw.githubusercontent.com/zyros-dev/pgx-lower/fix-bench-report-glob/benchmarks/pr-5-fix-bench-report-glob.png)

PG column is PostgreSQL-only runtime from the current run — reference for absolute pgx-lower cost, not part of the A/B.

| Query | PG (ms) | pgx baseline (ms) | pgx current (ms) | Δ | ✓ |
|-------|--------:|------------------:|-----------------:|--:|:--|
| q01 | 57.9 | 369.8 | 372.9 | -0.8% ⚪ | ✅ |
| q02 | 6.9 | 694.9 | 690.2 | +0.7% ⚪ | ✅ |
| q03 | 11.1 | 505.8 | 500.9 | +1.0% ⚪ | ✅ |
| q04 | 3.9 | 228.1 | 230.5 | -1.1% 🔴 | ✅ |
| q05 | 39.8 | 375.2 | 378.1 | -0.8% ⚪ | ✅ |
| q06 | 8.2 | 128.6 | 129.4 | -0.6% ⚪ | ✅ |
| q07 | 7.0 | 414.7 | 413.4 | +0.3% ⚪ | ✅ |
| q08 | 71.7 | 488.5 | 489.6 | -0.2% ⚪ | ✅ |
| q09 | 26.8 | 471.0 | 473.8 | -0.6% ⚪ | ✅ |
| q10 | 15.1 | 405.7 | 406.0 | -0.1% ⚪ | ✅ |
| q11 | 3.2 | 316.1 | 319.6 | -1.1% 🔴 | ✅ |
| q12 | 12.8 | 235.9 | 237.5 | -0.7% ⚪ | ✅ |
| q13 | 8.2 | 191.1 | 193.5 | -1.3% 🔴 | ✅ |
| q14 | 8.6 | 178.5 | 179.4 | -0.5% ⚪ | ✅ |
| q15 | 8.3 | 267.4 | 270.4 | -1.1% 🔴 | ✅ |
| q16 | 4.6 | 287.2 | 290.1 | -1.0% 🔴 | ✅ |
| q18 | 24.8 | 523.0 | 524.2 | -0.2% ⚪ | ✅ |
| q19 | 15.1 | 306.9 | 307.2 | -0.1% ⚪ | ✅ |
| q21 | 12.1 | 597.0 | 603.3 | -1.1% 🔴 | ✅ |
| q22 | 3.2 | 271.3 | 273.1 | -0.7% ⚪ | ✅ |
