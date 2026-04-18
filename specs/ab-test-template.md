# A/B test harness — shared template

Use this exact procedure for every spec's A/B section. Variations belong in the
spec, not here.

## Where to run

**Thor only.** macOS lacks perf stat and magic-trace. Per
`memory/project_thor_remote_build.md`:

- SSH to thor: `ssh comfy` (user `zel`)
- Repo path on thor: `/home/zel/repos/pgx-lower`
- Mac edits flow to thor via mutagen session `pgx-lower`

## Build the extension on thor

```bash
ssh comfy
cd ~/repos/pgx-lower/docker
docker compose up -d dev
docker exec pgx-lower-dev bash -c "
  cd /workspace &&
  mkdir -p build-docker-ptest &&
  cd build-docker-ptest &&
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY_EXTENSION=ON .. &&
  cmake --build .
"
```

Artifact: `build-docker-ptest/extension/pgx_lower.so`.

> **Use Release for perf measurements.** Debug builds skew vectorization,
> inlining, and IPC numbers significantly. Use Debug only when chasing a crash
> reproduced under the spec's own changes.

## Run the fast 4-query sweep (SF=0.01)

Default subset: **q01, q06, q03, q12** — covers scan+agg, single-table scan,
multi-way join, and conditional aggregate. ~2 minutes total per pgx_enabled
mode. The harness runs both modes per iteration.

```bash
ssh comfy 'cd ~/repos/pgx-lower &&
docker compose -f docker/docker-compose.yml up -d benchmark &&
python3 benchmark/tpch/run.py 0.01 \
  --port 54322 \
  --container benchmark \
  --skip "q02,q04,q05,q07,q08,q09,q10,q11,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22" \
  --iterations 3 \
  --label "<SPEC_ID>: <branch_name>"'
```

`--iterations 3` gives one cold (iter 1) + two warm (iter 2,3) runs per query
per mode. Always read both numbers.

## Pull the numbers

The harness writes to `benchmark/output/benchmark.db` (SQLite). For a side-by-side
table:

```bash
ssh comfy 'sqlite3 ~/repos/pgx-lower/benchmark/output/benchmark.db "
SELECT q.query_name,
       q.iteration,
       ROUND(AVG(CASE WHEN q.pgx_enabled=0
                      THEN json_extract(q.execution_metadata, ''\$.duration_ms'') END), 1) AS pg_ms,
       ROUND(AVG(CASE WHEN q.pgx_enabled=1
                      THEN json_extract(q.execution_metadata, ''\$.duration_ms'') END), 1) AS pgx_ms,
       ROUND(100.0 * (1.0 -
         AVG(CASE WHEN q.pgx_enabled=1 THEN json_extract(q.execution_metadata, ''\$.duration_ms'') END) /
         AVG(CASE WHEN q.pgx_enabled=0 THEN json_extract(q.execution_metadata, ''\$.duration_ms'') END)
       ), 1) AS speedup_pct
FROM queries q
JOIN runs r ON q.run_id = r.run_id
WHERE r.label LIKE ''<SPEC_ID>%''
GROUP BY q.query_name, q.iteration
ORDER BY q.query_name, q.iteration;
"'
```

`speedup_pct > 0` = pgx-lower faster. Negative = pgx-lower slower.

## Branch prediction / LLC numbers (when the spec asks for them)

Specs that change codegen quality (02, 09) should also capture perf counters:

```bash
ssh comfy 'cd ~/repos/pgx-lower &&
python3 benchmark/run_benchmark_config.py branch-prediction'
```

This runs the `branch-prediction` profile from `benchmark-config.yaml`
(SF=0.01, 100 iterations, perf stat enabled). Pull from the `perf_stats`
table:

```sql
SELECT query_name, pgx_enabled,
       AVG(branch_miss_rate) AS bm_pct,
       AVG(llc_miss_rate)    AS llc_pct,
       AVG(ipc)              AS ipc
FROM perf_stats
WHERE run_id LIKE '<SPEC_ID>%'
GROUP BY query_name, pgx_enabled;
```

## Baseline before you start

**Always.** Before touching code, run the harness on `main` and save the
resulting `benchmark.db` row IDs (or copy the file aside). Without a known-good
baseline, your delta is meaningless.

```bash
ssh comfy 'cp ~/repos/pgx-lower/benchmark/output/benchmark.db \
           ~/repos/pgx-lower/benchmark/output/baseline-$(date +%Y%m%d-%H%M).db'
```

## What goes in the PR description

1. Baseline numbers (4-query table, cold + warm).
2. Branch numbers (same table).
3. Speedup-percent column.
4. perf-stat counters if the spec listed them in its acceptance criteria.
5. One sentence: "shipped" or "neutral, kept for code-quality reasons" or
   "regression, here's why."

## Known harness gaps

- No built-in statistical significance test. With 3 iterations the noise is
  visible; if the delta is < 5% on warm runs, run with `--iterations 10` and
  eyeball variance before claiming a win or loss.
- No automatic baseline comparison — you save the baseline DB and diff manually.
- `enable_indexscan = off` is the default in the harness; don't flip it on
  unless the spec explicitly says so. Index-scan codegen is a separate path
  (see spec 09).
