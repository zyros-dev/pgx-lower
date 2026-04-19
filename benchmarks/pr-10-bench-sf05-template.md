## 🟡 BASELINE-SF-MISMATCH — unable to compare

- **Baseline scale factor:** 0.01
- **Current scale factor:**  0.5

![bench diff](https://raw.githubusercontent.com/zyros-dev/pgx-lower/bench-sf05-template/benchmarks/pr-10-bench-sf05-template.png)

> The baseline and this run used different TPC-H scale factors, so per-query percentages reflect data-size change (exec time scales linearly with SF), not code change. Verdict withheld. Once at least one PR at the current SF merges to main, future PRs will have a same-SF baseline.

| Query | PG (ms) | pgx baseline (ms) | pgx current (ms) | ✓ |
|-------|--------:|------------------:|-----------------:|:--|
| q01 | 908.3 | 567.3 | 5246.2 | ✅ |
| q02 | 126.7 | 579.2 | 2069.2 | ✅ |
| q03 | 285.0 | 502.3 | 6382.7 | ✅ |
| q04 | 99.8 | 228.7 | 2584.3 | ✅ |
| q05 | 117.5 | 380.7 | 3268.4 | ✅ |
| q06 | 148.7 | 129.6 | 2829.7 | ✅ |
| q07 | 162.9 | 404.0 | 3505.5 | ✅ |
| q08 | 249.7 | 490.5 | 3599.4 | ✅ |
| q09 | 571.8 | 477.0 | 6616.9 | ✅ |
| q10 | 266.5 | 387.2 | 5886.5 | ✅ |
| q11 | 56.5 | 318.0 | 1360.4 | ✅ |
| q12 | 231.8 | 234.3 | 3094.8 | ✅ |
| q13 | 600.3 | 192.9 | 1648.4 | ✅ |
| q14 | 145.2 | 182.7 | 2603.3 | ✅ |
| q15 | 143.2 | 260.2 | 2866.2 | ✅ |
| q16 | 142.5 | 288.8 | 2273.7 | ✅ |
| q18 | 1196.0 | 502.4 | 13257.1 | ✅ |
| q19 | 224.7 | 290.4 | 4337.3 | ✅ |
| q21 | 245.3 | 574.0 | 6938.6 | ✅ |
| q22 | 89.5 | 271.0 | 915.8 | ✅ |
