# pgx-lower Phase-Timing Report — SF=1.0

**Generated:** 2026-05-17T08:10:31  
**Scale Factor:** 1.0  
**Iterations:** 5  
**Skipped:** q17, q20  

## How to read this table

- **PG total** — median wall time with `pgx_lower.enabled=off` (plain PostgreSQL, indexes enabled).
- **pgxl total** — median wall time with `pgx_lower.enabled=on`.
- **compile** = setup + translate + lowering + jit (all median values, ms).
- **exec** = exec_ms from PGXL_PHASE_TIMING (median, ms).
- All times are medians over the reported iterations. `PGXL_PHASE_TIMING` is captured via `client_min_messages=debug1`.

## Per-Query Phase Split (SF=1.0)

| Query | PG total ms | pgxl total ms | setup ms | translate ms | lowering ms | jit ms | compile ms | exec ms | exec/compile ratio |
|-------|------------|---------------|----------|--------------|------------|--------|------------|---------|-------------------|
| q01 | 23857.9 | 23847.9 | 1.7 | 3.0 | 328.7 | 38.9 | 372.2 | 23409.2 | 62.89x |
| q02 | 6088.2 | 5994.3 | 1.8 | 4.2 | 386.2 | 63.0 | 455.2 | 5531.4 | 12.15x |
| q03 | 28228.0 | 28169.6 | 1.5 | 1.6 | 131.9 | 24.8 | 159.8 | 11644.5 | 72.88x |
| q04 | 11910.8 | 11889.1 | 1.6 | 1.6 | 133.2 | 24.4 | 160.9 | 11725.2 | 72.86x |
| q05 | 15240.1 | 15200.0 | 1.7 | 4.3 | 232.5 | 38.6 | 277.1 | 14904.7 | 53.78x |
| q06 | 14326.5 | 14286.7 | 1.7 | 1.8 | 56.0 | 9.5 | 69.0 | 14193.6 | 205.64x |
| q07 | 15653.9 | 15601.5 | 1.7 | 5.6 | 266.0 | 43.0 | 316.2 | 15260.4 | 48.26x |
| q08 | 15760.6 | 15689.9 | 1.8 | 8.8 | 327.2 | 48.6 | 386.4 | 15287.6 | 39.56x |
| q09 | 25526.0 | 25514.0 | 1.8 | 5.3 | 256.5 | 41.9 | 305.5 | 25197.6 | 82.47x |
| q10 | 23565.6 | 23551.5 | 1.6 | 3.5 | 253.0 | 45.1 | 303.3 | 23232.7 | 76.61x |
| q11 | 4067.8 | 4048.2 | 1.5 | 4.9 | 237.0 | 36.7 | 280.1 | 3760.4 | 13.42x |
| q12 | 14790.3 | 14849.0 | 1.6 | 3.8 | 152.7 | 23.1 | 181.2 | 14663.8 | 80.93x |
| q13 | 7767.7 | 7788.5 | 1.8 | 1.4 | 129.5 | 22.4 | 155.1 | 7626.6 | 49.18x |
| q14 | 12884.6 | 12904.4 | 1.6 | 3.7 | 100.6 | 15.5 | 121.6 | 12766.4 | 105.01x |
| q15 | 13004.0 | 13004.2 | 1.7 | 2.4 | 166.0 | 25.0 | 194.9 | 12803.1 | 65.68x |
| q16 | 5669.7 | 5662.7 | 2.0 | 1.7 | 153.2 | 27.8 | 184.6 | 5464.9 | 29.60x |
| q18 | 41161.4 | 41262.1 | 1.7 | 2.5 | 234.3 | 38.5 | 277.1 | 40962.4 | 147.83x |
| q19 | 21636.3 | 21697.9 | 1.7 | 6.1 | 145.4 | 24.5 | 177.7 | 21442.3 | 120.67x |
| q21 | 30793.1 | 30806.0 | 1.6 | 3.9 | 298.7 | 47.4 | 351.6 | 30456.8 | 86.62x |
| q22 | 3384.9 | 3363.5 | 1.5 | 2.3 | 212.9 | 34.2 | 250.9 | 3112.0 | 12.40x |

## Decision Gate: Is Execution Observable at SF=1.0?

Gate criterion: for Q01 and Q18, is `exec_ms` at least the same order of magnitude as `lowering_ms`? (i.e. execution is not dwarfed by compile)

**Q01:** lowering_ms=328.7 exec_ms=23409.2 compile_total=372.2 ratio=exec/lowering=71.2x — **YES**
**Q18:** lowering_ms=234.3 exec_ms=40962.4 compile_total=277.1 ratio=exec/lowering=174.8x — **YES**

(Go/no-go adjudicated by the controller, not this report.)

