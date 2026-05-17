#!/usr/bin/env python3
"""
Phase-timing benchmark for pgx-lower.

Runs all TPC-H queries (or a subset) at a given scale factor, capturing the
5-field PGXL_PHASE_TIMING log line emitted by pgx-lower per query.

The PGXL_PHASE_TIMING line format:
    PGXL_PHASE_TIMING setup_ms=N translate_ms=N lowering_ms=N jit_ms=N exec_ms=N query=N

It is emitted at DEBUG1 to the PostgreSQL log ONLY when these GUCs are set:
    pgx_lower.log_enable=on
    pgx_lower.log_io=on
    pgx_lower.enabled_categories='general'

We capture it via psycopg2's conn.notices list by setting client_min_messages=debug1.

Output:  benchmark/profiling/phase-timing/sf1-report.md   (by default)

Usage (inside container):
    python3 benchmark/profiling/phase-timing/run_phase_timing.py \\
        --sf 1 --port 5432 --iterations 5 --output benchmark/profiling/phase-timing/sf1-report.md

    # Single query, debugging:
    python3 benchmark/profiling/phase-timing/run_phase_timing.py \\
        --sf 1 --port 5432 --query q01 --iterations 3
"""

import argparse
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

import psycopg2


# ─── constants ────────────────────────────────────────────────────────────────

QUERIES_DIR = Path(__file__).parent.parent.parent / "tpch" / "queries"
# Queries to skip (same defaults as the main bench recipe):
DEFAULT_SKIP = {"q17", "q20"}

PHASE_RE = re.compile(
    r"PGXL_PHASE_TIMING\s+"
    r"setup_ms=([0-9.]+)\s+"
    r"translate_ms=([0-9.]+)\s+"
    r"lowering_ms=([0-9.]+)\s+"
    r"jit_ms=([0-9.]+)\s+"
    r"exec_ms=([0-9.]+)\s+"
    r"query=(\d+)"
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def pg_connect(port: int) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host="localhost", port=port, database="postgres", user="postgres"
    )
    # GUC SETs are session-scoped; they persist for subsequent statements on this connection.
    return conn


def enable_pgxl(cur) -> None:
    """Load pgx_lower and enable phase-timing GUCs."""
    cur.execute("LOAD 'pgx_lower.so'")
    cur.execute("SET pgx_lower.enabled = on")
    cur.execute("SET pgx_lower.log_enable = on")
    cur.execute("SET pgx_lower.log_io = on")
    cur.execute("SET pgx_lower.enabled_categories = 'general'")
    cur.execute("SET client_min_messages = debug1")


def disable_pgxl(cur) -> None:
    """Disable pgx_lower; use plain PostgreSQL."""
    cur.execute("SET pgx_lower.enabled = off")
    cur.execute("SET client_min_messages = warning")


def parse_phase_timing(notices: list[str]) -> dict | None:
    """
    Scan conn.notices for a PGXL_PHASE_TIMING line and return a dict with
    the 5 phase fields, or None if no matching line is found.
    """
    for msg in reversed(notices):  # the timing line is usually last
        m = PHASE_RE.search(msg)
        if m:
            return {
                "setup_ms":     float(m.group(1)),
                "translate_ms": float(m.group(2)),
                "lowering_ms":  float(m.group(3)),
                "jit_ms":       float(m.group(4)),
                "exec_ms":      float(m.group(5)),
                "query_id":     int(m.group(6)),
            }
    return None


def run_single_query(conn, query_sql: str, pgx: bool) -> dict:
    """
    Run one query and return timing data.
    Wall time is measured by the Python caller (psycopg2 round-trip).
    For pgx=True, phase timing is parsed from conn.notices.
    """
    conn.notices.clear()
    with conn.cursor() as cur:
        if pgx:
            enable_pgxl(cur)
        else:
            disable_pgxl(cur)
        conn.commit()

        conn.notices.clear()
        t0 = time.perf_counter()
        cur.execute(query_sql)
        cur.fetchall()  # consume result set
        wall_ms = (time.perf_counter() - t0) * 1000
        conn.commit()

    notices_snapshot = list(conn.notices)
    phase = parse_phase_timing(notices_snapshot) if pgx else None

    return {
        "wall_ms": wall_ms,
        "phase": phase,
    }


def load_query_files(query_filter: str | None, skip: set[str]) -> list[Path]:
    if not QUERIES_DIR.exists():
        sys.exit(f"ERROR: queries directory not found: {QUERIES_DIR}")

    files = sorted(QUERIES_DIR.glob("q*.sql"))
    if query_filter:
        files = [f for f in files if f.stem == query_filter]
        if not files:
            sys.exit(f"ERROR: query '{query_filter}' not found in {QUERIES_DIR}")

    return [f for f in files if f.stem not in skip]


def fmt_ms(v: float) -> str:
    return f"{v:.1f}"


# ─── report writer ────────────────────────────────────────────────────────────

def write_report(
    out_path: Path,
    sf: float,
    iterations: int,
    pg_results: dict[str, list[float]],
    pgxl_results: dict[str, list[dict]],
    skipped: set[str],
    run_ts: str,
) -> None:
    """Write the phase-timing report with per-query 5-phase split and gate evaluation."""

    queries = sorted(pg_results.keys())

    def med(vals):
        if not vals:
            return None
        return median(vals)

    def med_field(dicts, field):
        vals = [d[field] for d in dicts if d and field in d]
        if not vals:
            return None
        return median(vals)

    lines = []
    lines.append(f"# pgx-lower Phase-Timing Report — SF={sf}")
    lines.append(f"")
    lines.append(f"**Generated:** {run_ts}  ")
    lines.append(f"**Scale Factor:** {sf}  ")
    lines.append(f"**Iterations:** {iterations}  ")
    lines.append(f"**Skipped:** {', '.join(sorted(skipped)) or 'none'}  ")
    lines.append("")
    lines.append("## How to read this table")
    lines.append("")
    lines.append(
        "- **PG total** — median wall time with `pgx_lower.enabled=off` (plain PostgreSQL, indexes enabled)."
    )
    lines.append(
        "- **pgxl total** — median wall time with `pgx_lower.enabled=on`."
    )
    lines.append(
        "- **compile** = setup + translate + lowering + jit (all median values, ms)."
    )
    lines.append(
        "- **exec** = exec_ms from PGXL_PHASE_TIMING (median, ms)."
    )
    lines.append(
        "- All times are medians over the reported iterations. "
        "`PGXL_PHASE_TIMING` is captured via `client_min_messages=debug1`."
    )
    lines.append("")

    # ── per-query table ──────────────────────────────────────────────────────
    lines.append("## Per-Query Phase Split (SF={sf})".format(sf=sf))
    lines.append("")
    hdr = (
        "| Query | PG total ms | pgxl total ms | setup ms | translate ms | "
        "lowering ms | jit ms | compile ms | exec ms | exec/compile ratio |"
    )
    sep = (
        "|-------|------------|---------------|----------|--------------|"
        "------------|--------|------------|---------|-------------------|"
    )
    lines.append(hdr)
    lines.append(sep)

    gate_q01 = None
    gate_q18 = None

    for q in queries:
        pg_wall = med(pg_results[q])
        pgxl_recs = pgxl_results[q]
        pgxl_wall = med([r["wall_ms"] for r in pgxl_recs])

        phases = [r["phase"] for r in pgxl_recs if r["phase"] is not None]

        setup    = med_field(phases, "setup_ms")
        trans    = med_field(phases, "translate_ms")
        lower    = med_field(phases, "lowering_ms")
        jit      = med_field(phases, "jit_ms")
        exec_ms  = med_field(phases, "exec_ms")

        if all(v is not None for v in (setup, trans, lower, jit, exec_ms)):
            compile_ms = setup + trans + lower + jit
            ratio = exec_ms / compile_ms if compile_ms > 0 else float("inf")
            row = (
                f"| {q} | {fmt_ms(pg_wall) if pg_wall else 'N/A'} "
                f"| {fmt_ms(pgxl_wall) if pgxl_wall else 'N/A'} "
                f"| {fmt_ms(setup)} | {fmt_ms(trans)} | {fmt_ms(lower)} "
                f"| {fmt_ms(jit)} | {fmt_ms(compile_ms)} | {fmt_ms(exec_ms)} "
                f"| {ratio:.2f}x |"
            )
            if q == "q01":
                gate_q01 = (lower, exec_ms, compile_ms)
            if q == "q18":
                gate_q18 = (lower, exec_ms, compile_ms)
        else:
            row = (
                f"| {q} | {fmt_ms(pg_wall) if pg_wall else 'N/A'} "
                f"| {fmt_ms(pgxl_wall) if pgxl_wall else 'N/A'} "
                f"| N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
        lines.append(row)

    lines.append("")

    # ── decision gate ────────────────────────────────────────────────────────
    lines.append("## Decision Gate: Is Execution Observable at SF={sf}?".format(sf=sf))
    lines.append("")
    lines.append(
        "Gate criterion: for Q01 and Q18, is `exec_ms` at least the same order of magnitude "
        "as `lowering_ms`? (i.e. execution is not dwarfed by compile)"
    )
    lines.append("")

    for label, gate in (("Q01", gate_q01), ("Q18", gate_q18)):
        if gate is None:
            lines.append(f"**{label}:** DATA MISSING — phase timing not captured")
            continue
        lower_ms, exec_ms_v, compile_ms = gate
        same_order = exec_ms_v >= lower_ms * 0.1  # within 1 order of magnitude
        verdict = "YES" if same_order else "NO"
        ratio = exec_ms_v / lower_ms if lower_ms > 0 else float("inf")
        lines.append(
            f"**{label}:** lowering_ms={lower_ms:.1f} exec_ms={exec_ms_v:.1f} "
            f"compile_total={compile_ms:.1f} ratio=exec/lowering={ratio:.1f}x — **{verdict}**"
        )
    lines.append("")
    lines.append("(Go/no-go adjudicated by the controller, not this report.)")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to: {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Capture PGXL_PHASE_TIMING per TPC-H query and write split report."
    )
    parser.add_argument("--sf", type=float, default=1.0, help="TPC-H scale factor (default 1.0)")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port (default 5432)")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per query (default 5)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Run only this query (e.g. q01)")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated queries to skip")
    parser.add_argument(
        "--output", "-o", type=str,
        default=None,
        help="Output report path (default: sf{sf}-report.md next to this script)",
    )
    parser.add_argument("--no-pg-baseline", action="store_true",
                        help="Skip PostgreSQL baseline (pgx-lower runs only)")
    args = parser.parse_args()

    # Derive default output path from the actual --sf value so that different
    # scale-factor runs don't overwrite each other.
    if args.output is None:
        sf_tag = int(args.sf) if args.sf == int(args.sf) else args.sf
        args.output = str(Path(__file__).parent / f"sf{sf_tag}-report.md")

    skip = DEFAULT_SKIP.copy()
    if args.skip:
        skip |= set(args.skip.split(","))

    query_files = load_query_files(args.query, skip)
    if not query_files:
        sys.exit("ERROR: no queries to run")

    print(f"Phase-timing benchmark: SF={args.sf}, {args.iterations} iterations")
    print(f"Queries: {[f.stem for f in query_files]}")
    print(f"Skipped: {sorted(skip)}")

    conn = pg_connect(args.port)

    # Verify TPC-H data is loaded — if not, delegate to tpch/run.py which
    # handles the full idempotent dbgen + psql load pipeline.
    expected = int(150_000 * args.sf)
    needs_load = False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('customer')")
            if cur.fetchone()[0] is None:
                needs_load = True
            else:
                cur.execute("SELECT COUNT(*) FROM customer")
                count = cur.fetchone()[0]
                needs_load = count != expected
        conn.commit()
    except Exception as e:
        conn.rollback()
        needs_load = True

    if needs_load:
        print(
            f"TPC-H data not loaded at SF={args.sf} (expected {expected} customers). "
            "Delegating to benchmark/tpch/run.py to load data..."
        )
        conn.close()
        run_py = Path(__file__).parent.parent.parent / "tpch" / "run.py"
        # Run just enough to trigger load, then exit; we re-run queries ourselves
        load_result = subprocess.run(
            [
                sys.executable, str(run_py),
                str(args.sf), "--port", str(args.port),
                "--container", "phase-timing-loader",
                # Skip all queries to just load the data
                "--query", "q01", "--iterations", "1",
            ],
            capture_output=False,
        )
        if load_result.returncode != 0:
            sys.exit("ERROR: TPC-H data load failed")
        conn = pg_connect(args.port)
    else:
        print(f"TPC-H data verified: {expected} customers (SF={args.sf})")

    # Enable indexes for benchmark (match main bench recipe)
    with conn.cursor() as cur:
        cur.execute("SET enable_indexscan = on")
        cur.execute("SET enable_bitmapscan = on")
    conn.commit()

    pg_results: dict[str, list[float]] = defaultdict(list)
    pgxl_results: dict[str, list[dict]] = defaultdict(list)

    # ── parser sanity check ──────────────────────────────────────────────────
    # Verify the regex on a known-good line before running anything
    test_line = "DEBUG:  [GENERAL:IO] mlir_runner.cpp:199: PGXL_PHASE_TIMING setup_ms=1.234 translate_ms=0.567 lowering_ms=123.456 jit_ms=22.100 exec_ms=8500.123 query=1\n"
    parsed = parse_phase_timing([test_line])
    assert parsed is not None, "BUG: parser failed on known-good PGXL_PHASE_TIMING line"
    assert abs(parsed["setup_ms"] - 1.234) < 0.001
    assert abs(parsed["exec_ms"] - 8500.123) < 0.001
    assert parsed["query_id"] == 1
    print("Parser self-test: PASSED")

    # ── PostgreSQL baseline ──────────────────────────────────────────────────
    if not args.no_pg_baseline:
        print(f"\n--- PostgreSQL baseline (pgx_lower.enabled=off) ---")
        for qf in query_files:
            query_sql = qf.read_text()
            qname = qf.stem
            for it in range(1, args.iterations + 1):
                print(f"  {qname} iter {it}/{args.iterations}...", end=" ", flush=True)
                try:
                    res = run_single_query(conn, query_sql, pgx=False)
                    pg_results[qname].append(res["wall_ms"])
                    print(f"{res['wall_ms']:.1f}ms")
                except Exception as e:
                    print(f"ERROR: {e}")
                    conn.rollback()

    # ── pgx-lower phase runs ─────────────────────────────────────────────────
    print(f"\n--- pgx-lower with phase timing (pgx_lower.enabled=on) ---")
    for qf in query_files:
        query_sql = qf.read_text()
        qname = qf.stem
        for it in range(1, args.iterations + 1):
            print(f"  {qname} iter {it}/{args.iterations}...", end=" ", flush=True)
            try:
                res = run_single_query(conn, query_sql, pgx=True)
                pgxl_results[qname].append(res)
                phase = res["phase"]
                if phase:
                    print(
                        f"{res['wall_ms']:.1f}ms  "
                        f"[setup={phase['setup_ms']:.1f} trans={phase['translate_ms']:.1f} "
                        f"lower={phase['lowering_ms']:.1f} jit={phase['jit_ms']:.1f} "
                        f"exec={phase['exec_ms']:.1f}]"
                    )
                else:
                    print(f"{res['wall_ms']:.1f}ms  [NO PHASE TIMING CAPTURED]")
            except Exception as e:
                print(f"ERROR: {e}")
                conn.rollback()

    conn.close()

    # ── write report ─────────────────────────────────────────────────────────
    out_path = Path(args.output)
    write_report(
        out_path=out_path,
        sf=args.sf,
        iterations=args.iterations,
        pg_results=dict(pg_results),
        pgxl_results=dict(pgxl_results),
        skipped=skip,
        run_ts=datetime.now().isoformat(timespec="seconds"),
    )


if __name__ == "__main__":
    main()
