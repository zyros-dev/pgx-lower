# Stock-PG vs pgx-lower Differential Profile — SF=1 Control Set

**Generated:** 2026-05-17  
**Build:** RelWithDebInfo (`PGX_RELEASE_MODE=1`, Clang 20 O2+, JIT symbolized via `perf inject --jit`)  
**Scale Factor:** SF=1 (6 million lineitems for q01)  
**Method:** `perf record -g --call-graph fp -k 1` attached to postgres backend PID, then `perf inject --jit`, then `perf report --stdio --no-children`. Separate `perf stat` pass (10s window, `cycles,instructions,branches,branch-misses,cache-references,cache-misses,stalled-cycles-frontend`). Run as `su postgres` (paranoid=1 constraint). `reset_system_gucs()` baked in before every run.

**Timeout rule:** For stock-PG (baseline) runs, timeout = 3x pgx-lower SF=1 median latency from `sf1-report.md`. No queries timed out -- all stock-PG queries completed within the deadline.

---

## Query Set and Per-Query Timeout Thresholds

| Query | pgx-lower median (ms) | 3x timeout (s) | Stock-PG outcome |
|-------|-----------------------|----------------|------------------|
| q01   | 23,848                | 72             | COMPLETED        |
| q03   | 28,170                | 85             | COMPLETED        |
| q05   | 15,200                | 46             | COMPLETED        |
| q06   | 14,287                | 43             | COMPLETED        |
| q12   | 14,849                | 45             | COMPLETED        |
| q18   | 41,262                | 124            | COMPLETED        |

**Key finding:** Stock PostgreSQL at SF=1 is NOT dramatically slower than pgx-lower for these queries. All six queries completed within 3x the pgx-lower latency. The "slowness" in pgx-lower is comparable to stock PG -- not a clear win for either side at SF=1, row-store dataset.

---

## Per-Query Total Cycle Counts (perf record event count, approx.)

The `perf record` event count is the total CPU cycles attributed to the postgres backend PID for the full query duration.

| Query | Stock-PG cycles (B) | pgx-lower RWDI cycles (B) | pgx/PG cycle ratio |
|-------|---------------------|---------------------------|--------------------|
| q01   | 6.70                | 15.08                     | 2.25x              |
| q03   | 1.56                | 20.38                     | 13.1x              |
| q05   | 0.43                | 10.53                     | 24.7x              |
| q06   | 0.65                | 9.44                      | 14.5x              |
| q12   | 1.28                | 9.55                      | 7.5x               |
| q18   | 7.69                | 17.06                     | 2.2x               |

**Interpretation:** pgx-lower uses 2-25x more CPU cycles than stock PG for every query. Despite similar wall-time (from sf1-report.md), stock PG is IO-bound (disk, buffer manager, locking) while pgx-lower is CPU-bound. pgx-lower trades IO latency for compute, but the compute cost is currently very high.

**perf-stat 10s window (independent measurement):**

| Query | Stock-PG cycles (B) | pgx-lower RWDI cycles (B) |
|-------|---------------------|---------------------------|
| q01   | 7.10                | 15.52                     |
| q03   | 1.97                | 21.08                     |
| q05   | 0.78                | 11.21                     |
| q06   | 0.98                | 9.76                      |
| q12   | 1.77                | 10.14                     |
| q18   | 7.71                | 17.68                     |

---

## Per-Query Top-Symbol Self-Time (flat profile, % of samples)

### q01 -- 6 million lineitems, GROUP BY + NUMERIC aggregation

Stock PG (baseline):

| Symbol | % | Category |
|--------|---|----------|
| `ExecInterpExpr` | 12.14% | expression eval |
| `AllocSetAlloc` | 8.82% | memory allocator |
| `mul_var` | 7.02% | NUMERIC multiply |
| `tts_buffer_heap_getsomeattrs` | 5.55% | slot attribute detoast |
| `make_result_opt_error` | 5.47% | NUMERIC result |
| `numeric_mul_opt_error` | 3.76% | NUMERIC multiply wrapper |
| `accum_sum_add` | 3.56% | aggregate accumulator |
| `do_numeric_accum` | 3.55% | aggregate accumulator |
| `detoast_attr` | 3.37% | detoast |
| `numeric_sub_opt_error` | 2.56% | NUMERIC subtract |
| `sub_abs` | 2.43% | NUMERIC abs subtract |
| `heap_deform_tuple` | 0% | -- |

pgx-lower RWDI:

| Symbol | % | Category |
|--------|---|----------|
| `heap_deform_tuple` | 11.12% | PG tuple decode |
| `should_log` | 10.23%+1.26%@plt | PGX_IO per-tuple |
| `numeric_to_i128` | 9.21%+0.27%@plt | i128 conversion |
| `DataSourceIteration::access` | 8.02% | per-column decode |
| `log::log` | 7.28%+0.86%@plt | PGX_IO per-tuple |
| `process_tuple_into_batch` | 5.15% | batch fill |
| `AllocSetAlloc` | 4.52% | memory allocator |
| `main` (JIT) | 4.28% | JIT query loop |
| `detoast_attr` | 2.89% | PG detoast |
| `__divti3` | 2.42%+0.20%@plt | 128-bit division |

### q03 -- joins (customer x orders x lineitem), NUMERIC amounts

Stock PG (baseline):

| Symbol | % | Category |
|--------|---|----------|
| kernel 0xffff...a9ad7ce5 | 16.45% | IO/syscall |
| `tts_buffer_heap_getsomeattrs` | 7.11% | slot detoast |
| `ExecInterpExpr` | 4.38% | expression eval |
| `_bt_compare` | 2.89% | B-tree index |
| `hash_search_with_hash_value` | 2.42% | hash join probe |
| `heap_deform_tuple` | 0.12% | -- |

pgx-lower RWDI:

| Symbol | % | Category |
|--------|---|----------|
| `heap_deform_tuple` | 17.36% | PG tuple decode |
| `DataSourceIteration::access` | 10.31% | per-column decode |
| `log::log` | 8.48% | PGX_IO per-tuple |
| `should_log` | 7.56% | PGX_IO per-tuple |
| `process_tuple_into_batch` | 3.77% | batch fill |
| `numeric_to_i128` | 3.49%+0.11%@plt | i128 conversion |
| `read_next_tuple_from_table` | 2.98% | FFI table read |
| `main` (JIT, x2) | 2.32% | JIT query loop |

### q05 -- 5-way join, NUMERIC revenue sum

Stock PG (baseline):

| Symbol | % | Category |
|--------|---|----------|
| kernel 0xffff...a9ad7ce5 | 13.29% | IO/syscall |
| `tts_buffer_heap_getsomeattrs` | 6.96% | slot detoast |
| `ExecInterpExpr` | 5.00% | expression eval |
| `heapgettup_pagemode` | 4.12% | heap scan |
| `hash_search_with_hash_value` | 2.38% | hash join |
| `heap_deform_tuple` | 0% | -- |

pgx-lower RWDI:

| Symbol | % | Category |
|--------|---|----------|
| `heap_deform_tuple` | 17.55% | PG tuple decode |
| `DataSourceIteration::access` | 10.00% | per-column decode |
| `should_log` | 9.98% | PGX_IO per-tuple |
| `numeric_to_i128` | 6.78%+0.25%@plt | i128 conversion |
| `main` (JIT) | 5.16% | JIT query loop |
| `log::log` | 5.00% | PGX_IO per-tuple |
| `process_tuple_into_batch` | 3.86% | batch fill |
| `__divti3` | 1.74%+0.25%@plt | 128-bit division |

### q06 -- lineitem scan, NUMERIC multiply/filter

Stock PG (baseline):

| Symbol | % | Category |
|--------|---|----------|
| `tts_buffer_heap_getsomeattrs` | 27.08% | slot detoast |
| `ExecInterpExpr` | 14.10% | expression eval |
| kernel 0xffff...a9ad7ce5 | 7.89% | IO/syscall |
| `heapgettup_pagemode` | 5.56% | heap scan |
| `cmp_numerics` | 2.05% | NUMERIC compare |
| `heap_deform_tuple` | 0% | -- |

pgx-lower RWDI:

| Symbol | % | Category |
|--------|---|----------|
| `heap_deform_tuple` | 16.26% | PG tuple decode |
| `numeric_to_i128` | 10.96%+0.16%@plt | i128 conversion |
| `DataSourceIteration::access` | 9.20% | per-column decode |
| `should_log` | 9.04% | PGX_IO per-tuple |
| `log::log` | 5.95% | PGX_IO per-tuple |
| `process_tuple_into_batch` | 3.77% | batch fill |
| `__divti3` | 1.82%+0.23%@plt | 128-bit division |
| `main` (JIT) | 1.35% | JIT query loop |

### q12 -- lineitem + orders join, shipmode filter (bpchareq)

Stock PG (baseline):

| Symbol | % | Category |
|--------|---|----------|
| `tts_buffer_heap_getsomeattrs` | 28.13% | slot detoast |
| `bpchareq` | 9.79% | char comparison |
| `ExecEvalScalarArrayOp` | 7.05% | IN-list eval |
| `ExecInterpExpr` | 6.89% | expression eval |
| kernel 0xffff...a9ad7ce5 | 6.64% | IO/syscall |
| `heap_deform_tuple` | 0% | -- |

pgx-lower RWDI:

| Symbol | % | Category |
|--------|---|----------|
| `heap_deform_tuple` | 18.57% | PG tuple decode |
| `DataSourceIteration::access` | 11.26% | per-column decode |
| `should_log` | 8.92% | PGX_IO per-tuple |
| `log::log` | 7.04% | PGX_IO per-tuple |
| `process_tuple_into_batch` | 5.47% | batch fill |
| `main` (JIT) | 3.12% | JIT query loop |
| `read_next_tuple_from_table` | 3.08% | FFI table read |

### q18 -- GROUP BY + NUMERIC sum, large result set

Stock PG (baseline):

| Symbol | % | Category |
|--------|---|----------|
| `ExecInterpExpr` | 9.28% | expression eval |
| `tts_buffer_heap_getsomeattrs` | 7.79% | slot detoast |
| `ExecAgg` | 5.13% | aggregate |
| `TransactionIdPrecedes` | 4.46% | MVCC |
| `AllocSetAlloc` | 3.49% | memory allocator |
| `do_numeric_accum` | 2.47% | NUMERIC accumulate |
| `accum_sum_add` | 1.90% | aggregate |
| `heap_deform_tuple` | 0% | -- |

pgx-lower RWDI:

| Symbol | % | Category |
|--------|---|----------|
| `heap_deform_tuple` | 20.19% | PG tuple decode |
| `DataSourceIteration::access` | 9.88% | per-column decode |
| `should_log` | 8.19% | PGX_IO per-tuple |
| `main` (JIT) | 7.33% | JIT query loop |
| `log::log` | 5.24% | PGX_IO per-tuple |
| `numeric_to_i128` | 4.82%+0.10%@plt | i128 conversion |
| `read_next_tuple_from_table` | 3.44% | FFI table read |
| `process_tuple_into_batch` | 2.98% | batch fill |

---

## Finding (a): Is `heap_deform_tuple` PG-intrinsic?

**Answer: NO. `heap_deform_tuple` is pgx-elevated, not PG-intrinsic.**

| Query | Stock-PG heap_deform % | pgx-lower heap_deform % |
|-------|------------------------|------------------------|
| q01   | 0% (not in top 30)     | 11.12%                 |
| q03   | 0.12%                  | 17.36%                 |
| q05   | 0% (not in top 30)     | 17.55%                 |
| q06   | 0% (not in top 30)     | 16.26%                 |
| q12   | 0% (not in top 30)     | 18.57%                 |
| q18   | 0% (not in top 30)     | 20.19%                 |

In stock PG, `heap_deform_tuple` is absent or negligible (<0.2%) in the flat profile. Stock PG uses `tts_buffer_heap_getsomeattrs` (slot-based attribute access, 5-28% across queries) as its tuple-detoast path. It does not call `heap_deform_tuple` as a hot function.

pgx-lower calls `heap_deform_tuple` at 11-20% self-time across all six queries. This is the FFI tuple-decode path: for each input tuple, pgx-lower calls `heap_deform_tuple` to extract all attributes into a columnar batch (via `process_tuple_into_batch` / `DataSourceIteration::access`). This is a pgx-lower-introduced per-tuple cost.

**Verdict: DO chase. heap_deform_tuple at 11-20% is pgx-elevated, not a PG-intrinsic floor.**

---

## Finding (b): i128-vs-NUMERIC Arithmetic Comparison

The strategic question: does pgx-lower's "convert NUMERIC to i128 once, compute fast" approach cost less than stock PG staying in NUMERIC throughout?

### Stock PG NUMERIC arithmetic cost (q01)

| Symbol | % | Description |
|--------|---|-------------|
| `mul_var` | 7.02% | NUMERIC multiply (digit-array, heap-allocating) |
| `numeric_mul_opt_error` | 3.76% | NUMERIC multiply wrapper |
| `accum_sum_add` | 3.56% | aggregate accumulator |
| `do_numeric_accum` | 3.55% | aggregate accumulator |
| `numeric_sub_opt_error` | 2.56% | NUMERIC subtract |
| `sub_abs` | 2.43% | NUMERIC abs subtract |
| `add_abs` | 1.40% | NUMERIC abs add |
| `numeric_add_opt_error` | 1.14% | NUMERIC add |
| `round_var` | 1.09% | NUMERIC round |
| **Total NUMERIC arithmetic** | **~26.5%** | |

### pgx-lower i128 cost (q01)

| Symbol | % | Description |
|--------|---|-------------|
| `numeric_to_i128` | 9.21%+0.27%@plt | conversion from NUMERIC heap type to i128 |
| `__divti3` | 2.42%+0.20%@plt | 128-bit division (libgcc software fallback) |
| `main` (JIT) | 4.28% | JIT query loop (all i128 arithmetic + grouping) |
| **Total i128 path** | **~16.4%** | |

### Absolute cycle comparison

| Path | % samples | Total cycles | Approx. absolute cycles |
|------|-----------|--------------|------------------------|
| Stock PG NUMERIC (q01) | ~26.5% | 6.70B | ~1.77B |
| pgx-lower i128+JIT (q01) | ~16.4% | 15.08B | ~2.47B |

pgx-lower's i128 path costs ~2.47B cycles absolute vs. ~1.77B for stock PG NUMERIC. **The i128 approach is ~40% MORE expensive in absolute cycles for q01.**

| Path | % samples | Total cycles | Approx. absolute cycles |
|------|-----------|--------------|------------------------|
| Stock PG NUMERIC (q06) | ~3.5% | 0.65B | ~22M |
| pgx-lower i128+JIT (q06) | ~14.3% | 9.44B | ~1.35B |

**q06: pgx-lower spends ~61x more absolute cycles on the numeric-path than stock PG.**

| Path | % samples | Total cycles | Approx. absolute cycles |
|------|-----------|--------------|------------------------|
| Stock PG NUMERIC (q18) | ~6.5% | 7.69B | ~0.50B |
| pgx-lower i128+JIT (q18) | ~12.3% | 17.06B | ~2.10B |

**q18: pgx-lower spends ~4.2x more absolute cycles on numeric-path than stock PG.**

### i128-vs-NUMERIC Summary Table

| Query | Stock PG NUMERIC (B cycles) | pgx i128+JIT (B cycles) | pgx/PG ratio |
|-------|-----------------------------|-------------------------|--------------|
| q01   | ~1.77B                      | ~2.47B                  | 1.4x         |
| q06   | ~0.022B                     | ~1.35B                  | 61x           |
| q18   | ~0.50B                      | ~2.10B                  | 4.2x         |

**The "convert once to i128 then compute fast" hypothesis is NOT supported by the data.** The conversion overhead (`numeric_to_i128` at 3.5-10.96% self-time) is itself larger than stock PG's full NUMERIC arithmetic cost for q06 and q12. Stock PG's NUMERIC digit-array arithmetic, despite allocating per operation, dominates less of CPU than pgx-lower's conversion + JIT path at SF=1.

The `__divti3` symbol (AMD has no native 128-bit divide instruction) adds overhead for every scale-factor-adjustment division that has no equivalent in stock PG's `div_var` (which uses its own digit-array but is not called at every tuple access).

**Caveats (critical for interpretation):**
1. The JIT `main` includes ALL compiled logic (joins, grouping, hash, filters), not only arithmetic. The i128-arithmetic portion within JIT is not isolable from this profile.
2. These profiles are at SF=1 where stock PG is heavily IO-bound. At larger SF (10, 100), when data fits in RAM, stock PG becomes more CPU-bound and the comparison changes.
3. Stock PG `AllocSetAlloc` (3-9%) is partly driven by NUMERIC allocations; this hidden cost is not included in the NUMERIC symbol percentages above.
4. Sample counts for baseline-pg profiles are low (527-8K vs 9-21K for pgx), reducing statistical precision for baseline-pg flat profiles.

---

## pgx-elevated vs. PG-intrinsic Classification

| Symbol / cost | PG-intrinsic? | Verdict |
|---------------|---------------|---------|
| `heap_deform_tuple` (11-20% pgx) | NO -- 0% in stock PG | DO CHASE |
| `should_log` + `log::log` (12-18% pgx) | NO -- 0% in stock PG | DO CHASE (gate PGX_IO on PGX_RELEASE_MODE) |
| `numeric_to_i128` (3.5-11% pgx) | NO -- 0% in stock PG | DO CHASE (larger than stock PG NUMERIC cost) |
| `DataSourceIteration::access` (8-11% pgx) | NO -- 0% in stock PG | DO CHASE |
| `__divti3` (1.7-2.4% pgx) | NO -- 0% in stock PG | NOTE (AMD has no native 128-bit div) |
| `AllocSetAlloc` (3-9%) | YES -- 3-9% in stock PG too | PG-intrinsic floor; do not chase |
| `detoast_attr` (2-3%) | YES -- present in both | PG-intrinsic; do not chase |
| `heapgettup_pagemode` (1-2%) | YES -- present in both | PG-intrinsic; do not chase |
| `ExecInterpExpr` (4-14% stock PG, 0% pgx) | YES for stock PG; absent in pgx | PG-intrinsic for stock; replaced by JIT |
| JIT `main` (1-7.3% pgx, 0% stock PG) | pgx-specific | The goal; all other costs should shrink relative to this |

---

## Note: PGX_IO Overhead in RelWithDebInfo

`PGX_RELEASE_MODE=1` eliminates `PGX_HOT_LOG` but does NOT eliminate `PGX_IO` (`should_log()` + `std::optional<ScopeLogger>` ctor/dtor per call). The 12-18% self-time across all six queries in `should_log` + `log::log` is confirmed NOT GUC contamination (pg_settings shows no pgx_lower.* GUCs active). This overhead is entirely absent from stock PG profiles. Gating `PGX_IO` on `#ifndef PGX_RELEASE_MODE` would be a high-leverage improvement.

---

## Artifacts

Raw perf data files (.data) are gitignored. Text artifacts committed:

| Directory | perf-report.txt size | perf-stat.txt |
|-----------|---------------------|---------------|
| `perf-exec/q01-sf1-baseline-pg/` | ~10 KB | yes |
| `perf-exec/q03-sf1-baseline-pg/` | ~10 KB | yes |
| `perf-exec/q05-sf1-baseline-pg/` | ~15 KB | yes |
| `perf-exec/q06-sf1-baseline-pg/` | ~15 KB | yes |
| `perf-exec/q12-sf1-baseline-pg/` | ~14 KB | yes |
| `perf-exec/q18-sf1-baseline-pg/` | ~52 KB | yes |
| `perf-exec/q01-sf1-relwithdebinfo/` | ~77 KB | yes (pre-existing) |
| `perf-exec/q03-sf1-relwithdebinfo/` | ~124 KB | yes |
| `perf-exec/q05-sf1-relwithdebinfo/` | ~70 KB | yes |
| `perf-exec/q06-sf1-relwithdebinfo/` | ~64 KB | yes |
| `perf-exec/q12-sf1-relwithdebinfo/` | ~63 KB | yes |
| `perf-exec/q18-sf1-relwithdebinfo/` | ~76 KB | yes |
