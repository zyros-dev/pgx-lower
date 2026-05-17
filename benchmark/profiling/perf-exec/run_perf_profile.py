#!/usr/bin/env python3
"""
perf-profile runner for pgx-lower.

Profiles a single TPC-H query under pgx_lower.enabled=on using:
  1. perf record -g --call-graph dwarf  (sampled call-graph)
  2. perf stat -e <counters>            (hardware counter snapshot)

Artifacts are written to benchmark/profiling/perf-exec/<experiment>/:
  perf.data          — raw sampling data (perf record)
  perf-stat.txt      — hardware counter summary (perf stat)
  perf-report.txt    — perf report --stdio, top symbols

Usage (inside the pgx-lower-dev container):
    python3 benchmark/profiling/perf-exec/run_perf_profile.py \\
        --query q01 --sf 1 --port 5432 \\
        --output benchmark/profiling/perf-exec/q01-sf1

Requires:
  - perf binary at /usr/local/bin/perf (installed in container by hand or
    baked into the Dockerfile; see benchmark/profiling/tooling.md)
  - kernel.perf_event_paranoid <= 1 on the host
    (docker-compose.yml has cap_add: [SYS_ADMIN, PERFMON] + seccomp:unconfined)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psycopg2


QUERIES_DIR = Path(__file__).parent.parent.parent / "tpch" / "queries"

PERF_STAT_EVENTS = (
    "cycles,instructions,branches,branch-misses,"
    "cache-references,cache-misses,"
    "stalled-cycles-frontend,stalled-cycles-backend"
)


def pg_connect(port: int) -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host="localhost", port=port, database="postgres", user="postgres"
    )


def ensure_tpch_loaded(port: int, sf: float) -> None:
    """Verify TPC-H is loaded at the given SF; delegate to tpch/run.py if not."""
    expected = int(150_000 * sf)
    try:
        conn = pg_connect(port)
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('customer')")
            row = cur.fetchone()
            if row and row[0] is not None:
                cur.execute("SELECT COUNT(*) FROM customer")
                count = cur.fetchone()[0]
                if count == expected:
                    print(f"TPC-H data verified: {count} customers (SF={sf})")
                    conn.close()
                    return
        conn.close()
    except Exception as e:
        print(f"DB check error: {e}")

    print(f"TPC-H not loaded at SF={sf}. Delegating to tpch/run.py ...")
    run_py = QUERIES_DIR.parent / "run.py"
    result = subprocess.run(
        [
            sys.executable, str(run_py),
            str(sf), "--port", str(port),
            "--container", "pgx-lower-dev",
            "--query", "q01", "--iterations", "1",
        ],
        check=False,
    )
    if result.returncode != 0:
        sys.exit("ERROR: TPC-H data load failed")


def check_perf(perf_bin: str) -> None:
    """Verify perf binary is available and perf_event_paranoid is not blocking."""
    if not shutil.which(perf_bin) and not Path(perf_bin).exists():
        sys.exit(
            f"ERROR: perf binary not found at '{perf_bin}'.\n"
            "Install with: docker cp /tmp/linux-tools-6.8/usr/lib/linux-tools-6.8.0-31/perf"
            " pgx-lower-dev:/usr/local/bin/perf"
        )

    paranoid_path = "/proc/sys/kernel/perf_event_paranoid"
    try:
        paranoid = int(Path(paranoid_path).read_text().strip())
        if paranoid > 1:
            sys.exit(
                f"ERROR: kernel.perf_event_paranoid={paranoid} blocks perf_event_open.\n"
                "Fix (requires sudo on thor):\n"
                "  sudo sysctl -w kernel.perf_event_paranoid=1\n"
                "  echo 'kernel.perf_event_paranoid=1' | sudo tee /etc/sysctl.d/99-perf.conf\n"
                "See benchmark/profiling/tooling.md for details."
            )
    except Exception:
        pass  # /proc not readable inside some containers — let perf fail naturally


def run_query_sql(port: int, query_sql: str) -> None:
    """Execute the query via psycopg2 (for warm-up or baseline)."""
    conn = pg_connect(port)
    with conn.cursor() as cur:
        cur.execute("LOAD 'pgx_lower.so'")
        cur.execute("SET pgx_lower.enabled = on")
        t0 = time.perf_counter()
        cur.execute(query_sql)
        cur.fetchall()
        elapsed = time.perf_counter() - t0
    conn.close()
    print(f"  warm-up: {elapsed * 1000:.0f} ms")


def build_psql_cmd(port: int, query_sql_path: Path) -> list[str]:
    """
    Build the psql invocation that executes the query once with pgx_lower enabled.
    We wrap this entire command under perf so the profile covers the backend work.
    """
    # Set GUCs via -c flags before executing the file
    return [
        "psql",
        "-h", "localhost",
        "-p", str(port),
        "-U", "postgres",
        "-d", "postgres",
        "-c", "LOAD 'pgx_lower.so'",
        "-c", "SET pgx_lower.enabled = on",
        "-f", str(query_sql_path),
        "-c", "SELECT 1",   # force result consumption
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a TPC-H query under pgx_lower using perf record + perf stat."
    )
    parser.add_argument("--query", "-q", required=True, help="Query name, e.g. q01")
    parser.add_argument("--sf", type=float, default=1.0, help="TPC-H scale factor (default 1.0)")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL port (default 5432)")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory (default: benchmark/profiling/perf-exec/<query>-sf<sf>)",
    )
    parser.add_argument(
        "--perf-bin", default="/usr/local/bin/perf",
        help="Path to perf binary (default /usr/local/bin/perf)",
    )
    parser.add_argument("--skip-warmup", action="store_true", help="Skip warm-up query run")
    args = parser.parse_args()

    check_perf(args.perf_bin)

    query_sql_path = QUERIES_DIR / f"{args.query}.sql"
    if not query_sql_path.exists():
        sys.exit(f"ERROR: query file not found: {query_sql_path}")

    if args.output is None:
        sf_tag = int(args.sf) if args.sf == int(args.sf) else args.sf
        args.output = str(
            Path(__file__).parent / f"{args.query}-sf{sf_tag}"
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    perf_data = str(out_dir / "perf.data")
    perf_stat_txt = str(out_dir / "perf-stat.txt")
    perf_report_txt = str(out_dir / "perf-report.txt")

    ensure_tpch_loaded(args.port, args.sf)

    query_sql = query_sql_path.read_text()
    psql_cmd = build_psql_cmd(args.port, query_sql_path)

    if not args.skip_warmup:
        print(f"\n=== Warm-up run ({args.query} @ SF={args.sf}) ===")
        run_query_sql(args.port, query_sql)

    # ── perf record ──────────────────────────────────────────────────────────
    print(f"\n=== perf record ({args.query} @ SF={args.sf}) ===")
    print(f"  output: {perf_data}")
    record_cmd = [
        args.perf_bin,
        "record",
        "-g",
        "--call-graph", "dwarf",
        "-o", perf_data,
        "--",
    ] + psql_cmd

    t0 = time.perf_counter()
    rec_result = subprocess.run(record_cmd, capture_output=False)
    elapsed = time.perf_counter() - t0
    print(f"  elapsed: {elapsed * 1000:.0f} ms, exit: {rec_result.returncode}")

    if rec_result.returncode != 0:
        sys.exit(
            f"ERROR: perf record failed (exit {rec_result.returncode}).\n"
            "Check benchmark/profiling/tooling.md for the perf_event_paranoid fix."
        )

    size_bytes = Path(perf_data).stat().st_size if Path(perf_data).exists() else 0
    print(f"  perf.data size: {size_bytes:,} bytes")

    # ── perf stat ────────────────────────────────────────────────────────────
    print(f"\n=== perf stat ({args.query} @ SF={args.sf}) ===")
    print(f"  output: {perf_stat_txt}")
    stat_cmd = [
        args.perf_bin,
        "stat",
        "-e", PERF_STAT_EVENTS,
        "--",
    ] + psql_cmd

    with open(perf_stat_txt, "w") as stat_out:
        stat_out.write(
            f"# perf stat: {args.query} @ SF={args.sf}\n"
            f"# command: {' '.join(stat_cmd)}\n\n"
        )
        stat_result = subprocess.run(stat_cmd, stderr=subprocess.STDOUT, stdout=stat_out)

    print(f"  exit: {stat_result.returncode}")
    if stat_result.returncode == 0:
        print(f"  perf-stat.txt written ({Path(perf_stat_txt).stat().st_size:,} bytes)")

    # ── perf report ──────────────────────────────────────────────────────────
    if Path(perf_data).exists() and size_bytes > 0:
        print(f"\n=== perf report ({args.query} @ SF={args.sf}) ===")
        report_cmd = [
            args.perf_bin,
            "report",
            "--stdio",
            "-i", perf_data,
            "--no-children",
        ]
        with open(perf_report_txt, "w") as report_out:
            report_result = subprocess.run(
                report_cmd, stdout=report_out, stderr=subprocess.STDOUT
            )
        print(f"  exit: {report_result.returncode}")
        print(f"  perf-report.txt written ({Path(perf_report_txt).stat().st_size:,} bytes)")

        # Show top symbols + check for FFI symbols
        print("\n=== Top symbols ===")
        ffi_symbols = ["extract_field", "get_int32_field_mlir", "get_int64_field_mlir",
                       "get_date_field_mlir", "get_decimal_field_mlir", "get_float8_field_mlir"]
        with open(perf_report_txt) as f:
            content = f.read()

        # Print top 30 non-comment lines
        lines = [l for l in content.splitlines() if not l.startswith("#") and l.strip()]
        for line in lines[:30]:
            print(" ", line)

        print("\n=== FFI symbol check ===")
        found = [sym for sym in ffi_symbols if sym in content]
        if found:
            print(f"  FOUND: {found}")
        else:
            print("  NOT FOUND: no extract_field / get_*_field_mlir in perf report")
            print("  (This may mean the workload is not FFI-bound, or symbols need --demangle.)")

    print(f"\n=== Artifacts written to: {out_dir} ===")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
