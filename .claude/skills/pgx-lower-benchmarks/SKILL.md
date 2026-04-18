---
name: pgx-lower-benchmarks
description: The TPC-H benchmark harness for A/B testing pgx-lower vs PostgreSQL. SF=0.01 fast-iteration runs, perf-stat profiles, magic-trace, the SQLite results DB, profile catalog. Use when measuring performance, running A/B tests for a spec, adding benchmark profiles, or interpreting benchmark output.
---

# Benchmarks

The harness in `benchmark/` runs TPC-H queries through both pgx-lower and
vanilla PG and stores per-query timings + perf counters in a SQLite database.
Profiles are defined in `benchmark-config.yaml`.

**Run benchmarks on thor only.** macOS lacks `perf stat` and `magic-trace`.

## Files

```
benchmark-config.yaml         Profile definitions
benchmark/
├── run_benchmark_config.py   Profile orchestrator (entry point)
├── tpch/
│   ├── run.py                Per-run query executor + metrics
│   ├── aggregate.py          Statistics over runs (mean, p95, p99)
│   ├── metrics_collector.py  EXPLAIN ANALYZE BUFFERS extraction
│   ├── fxt_to_flamegraph.py  magic-trace FXT → flamegraph JSON
│   ├── queries/              q01.sql … q22.sql
│   └── requirements.txt      psycopg2, psutil, lz4, pyyaml
├── generate_tpch_data.py     dbgen wrapper, produces tpch_init.sql
├── tpch_init.sql             Pre-generated SF=0.01 data (~68MB, checked in)
└── output/
    ├── benchmark.db          Primary SQLite results DB
    ├── archive/              Auto-archived runs
    └── traces/               magic-trace outputs
```

## Profiles in benchmark-config.yaml

| Profile | SF | Iter | perf_stats | profile | Skipped queries | Use case |
|---------|----|----|-----------|---------|-----------------|----------|
| `quick` | 0.01 | 1 | no | no | none | Smoke test, ~1 minute |
| `profile` | 0.01 | varies | no | yes (magic-trace) | none | CPU profiling |
| `profile2` | 0.01 | varies | no | yes | varies | Index-issue investigation |
| `branch-prediction` | 0.01 | 100 | yes | no | q02,q17,q20,q21 | Hardware counters |
| `full` | 1 | varies | yes | varies | q07,q20 | Big-data sweep (slow) |

`q02, q17, q20, q21` are skipped in heavy profiles because they timeout or
crash on vanilla PG.

## Running

### Quick smoke test

```bash
ssh comfy 'cd ~/repos/pgx-lower &&
docker compose -f docker/docker-compose.yml up -d benchmark &&
python3 benchmark/run_benchmark_config.py quick'
```

### Direct run with custom flags

```bash
ssh comfy 'cd ~/repos/pgx-lower &&
python3 benchmark/tpch/run.py 0.01 \
  --port 54322 \
  --container benchmark \
  --label "my experiment" \
  --iterations 3'
```

### Branch-prediction / LLC profile

```bash
ssh comfy 'cd ~/repos/pgx-lower &&
python3 benchmark/run_benchmark_config.py branch-prediction'
```

Adds `perf stat -e task-clock,page-faults,cycles,instructions,branches,
branch-misses,LLC-loads,LLC-load-misses` per query and stores in `perf_stats`
table. ~5–15 minutes.

### Fast 4-query sweep (recommended for spec A/B)

Suggested subset: q01 (scan+agg), q06 (single-table scan), q03 (multi-way join),
q12 (conditional agg). ~2 minutes per pgx_enabled mode.

```bash
ssh comfy 'cd ~/repos/pgx-lower &&
python3 benchmark/tpch/run.py 0.01 \
  --port 54322 \
  --container benchmark \
  --skip "q02,q04,q05,q07,q08,q09,q10,q11,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22" \
  --iterations 3 \
  --label "my-spec-id: branch_name"'
```

## How pgx-enabled vs PG comparison works

Each iteration runs each query **twice**:
1. With `pgx_lower.enabled = false` → vanilla PG executor.
2. With `pgx_lower.enabled = true` → pgx-lower JIT path.

Stored in the same row pair in `queries` table, distinguished by
`pgx_enabled` BOOLEAN.

The harness validates correctness by hashing the first 5000 result rows per
query — both runs must produce the same hash, else the row is flagged
`result_valid = false`.

## Output: SQLite schema

`benchmark/output/benchmark.db`:

| Table | Notable columns |
|-------|-----------------|
| `runs` | run_id, timestamp, SF, iterations, PG version, pgx version, hostname, container, label |
| `queries` | run_id, query_name, iteration, pgx_enabled, execution_metadata (JSON), metrics_json (JSON), postgres_metrics (JSON), timeseries_metrics (JSON), result_validation (JSON) |
| `profiling` | query_id, cpu_data_lz4, heap_data_lz4, flamegraph_lz4 (blobs) |
| `perf_stats` | query_name, pgx_enabled, iteration, branch_miss_rate, llc_miss_rate, ipc, ... |
| `aggregate_benchmarks` | mean / p95 / p99 / min / max per config |
| `aggregate_profiles` | merged profiling per config |

## Pulling A/B numbers (paste-ready SQL)

```bash
ssh comfy 'sqlite3 ~/repos/pgx-lower/benchmark/output/benchmark.db "
SELECT q.query_name, q.iteration,
       ROUND(AVG(CASE WHEN q.pgx_enabled=0
              THEN json_extract(q.execution_metadata, ''\$.duration_ms'') END), 1) AS pg_ms,
       ROUND(AVG(CASE WHEN q.pgx_enabled=1
              THEN json_extract(q.execution_metadata, ''\$.duration_ms'') END), 1) AS pgx_ms,
       ROUND(100.0 * (1.0 -
         AVG(CASE WHEN q.pgx_enabled=1 THEN json_extract(q.execution_metadata, ''\$.duration_ms'') END) /
         AVG(CASE WHEN q.pgx_enabled=0 THEN json_extract(q.execution_metadata, ''\$.duration_ms'') END)
       ), 1) AS speedup_pct
FROM queries q JOIN runs r ON q.run_id = r.run_id
WHERE r.label LIKE ''<MY_LABEL>%''
GROUP BY q.query_name, q.iteration ORDER BY q.query_name, q.iteration;
"'
```

`speedup_pct > 0` = pgx-lower faster. Negative = slower.

For perf counters:

```sql
SELECT query_name, pgx_enabled,
       AVG(branch_miss_rate) AS bm_pct,
       AVG(llc_miss_rate)    AS llc_pct,
       AVG(ipc)              AS ipc
FROM perf_stats
WHERE run_id LIKE '<MY_LABEL>%'
GROUP BY query_name, pgx_enabled;
```

## TPC-H data

`benchmark/tpch_init.sql` is checked in (~68MB) for SF=0.01. For larger SF:

- `python3 benchmark/generate_tpch_data.py --sf 1.0` invokes dbgen (downloaded
  from GitHub commit 32f1c1b) and produces a fresh `tpch_init.sql`.
- ~2-5 min generation time at SF=0.01, much longer at SF=1.

Loading is via `psql -f tpch_init.sql` from `run.py:load_tpch_database()`.

Tables: `region, nation, part, supplier, partsupp, customer, orders, lineitem`.

## GUCs the harness toggles

- `pgx_lower.enabled` — flips between vanilla PG and pgx-lower per-pass.
- `enable_indexscan` and `enable_bitmapscan` — turned on/off via the
  benchmark config's `indexes` flag (`run.py:init_databases()` line 696-701).
  Default in most profiles: **off**. Index codegen is a separate path.

## Profile output (magic-trace)

When `profile: true` in the YAML profile:
- magic-trace runs in LBR (Intel) or stack-walk (AMD) mode.
- Output: FXT binary trace + lz4-compressed flamegraph JSON.
- Stored as blobs in `profiling` table.
- Symlinked under `output/traces/sf_<SF>_<timestamp>/`.

Convert to flamegraph for inspection:
```bash
python3 benchmark/tpch/fxt_to_flamegraph.py <FXT_PATH>
```

## Recommended A/B procedure (for any spec)

1. **Baseline first.** Always. Save a copy of the DB:
   ```bash
   ssh comfy 'cp ~/repos/pgx-lower/benchmark/output/benchmark.db \
              ~/repos/pgx-lower/benchmark/output/baseline-$(date +%Y%m%d-%H%M).db'
   ```
2. Build Release on the baseline branch.
3. Run the 4-query sweep with `--label "baseline"`.
4. Switch to the change branch, build Release.
5. Run the same sweep with `--label "<spec-id>: <branch>"`.
6. Pull the speedup-pct table.
7. If perf-counter changes are expected (codegen quality), run the
   `branch-prediction` profile too.
8. Paste the table and counters into the PR description.

## Known harness gaps

- **No automated A/B comparison tool.** SQL queries (above) compare manually.
- **Single-shot per query** — no built-in warmup. Use `--iterations 3+` to
  separate cold (iter 1) from warm (iter 2,3).
- **Magic-trace requires Linux kernel** (Intel PT or AMD sampling). macOS
  cannot run the profile profiles.
- **Indexes are toggled but not created.** The harness assumes indexes exist
  in the schema; default schema doesn't have them. The `8b912c2 enable
  indexes` commit added the toggle but the actual `CREATE INDEX` commands
  must come from the SQL setup. Verify before running with indexes on.
- **No JSON export.** Only SQLite + CSV. Adapt SQL queries above for
  external tooling.

## Lessons from the harness's own history

The benchmark infra had multiple revert thrashes (Oct 2025):
- 10 reverts/reapplies of dry-run scale factor configs — symptom of
  experimenting on master instead of a branch.
- Hardware counters (commit b210fe5) were the unlock for understanding why
  pgx-lower looked flat on cold runs — branch misprediction and LLC misses
  dominated.
- Index discovery (commits 8b912c2, 0a30144) revealed that benchmark stability
  depends on plan-shape stability; index/no-index plans diverge wildly.

Lesson: **when benchmarks look weird, add hardware counters before assuming
your code is the bug.**

## Related skills

- `pgx-lower-build-and-test` — produces the `.so` the harness loads.
- `pgx-lower-execution-path` — what `pgx_lower.enabled = on` actually
  switches.
- `specs/ab-test-template.md` — the canonical A/B procedure for spec PRs.
