#!/usr/bin/env python3
"""
perf-profile runner for pgx-lower.

Profiles a single TPC-H query under pgx_lower.enabled=on using:
  1. perf record -g --call-graph dwarf  (sampled call-graph, attached to the
     postgres BACKEND process — NOT the psql client)
  2. perf stat -e <counters>            (hardware counter snapshot, also on backend)

Why attach to the backend?
  psql is a thin client that is idle on the socket while the backend does all
  the work.  Profiling the psql process shows zero useful samples.  We must
  attach perf to the postgres backend PID.

How it works:
  1. Open a psycopg2 connection, load pgx_lower, discover the backend PID via
     SELECT pg_backend_pid().
  2. Launch the query on a background thread (it will block until complete).
  3. Attach perf record -p <backend_pid> as the postgres OS user via
     su postgres -c "perf record -p ... -o ..."
     (paranoid=1 blocks cross-user attach; postgres must own the perf process).
  4. Wait for the query thread to finish, then stop perf (SIGINT after join).
  5. Generate perf report --stdio from the captured perf.data.
  6. Run perf stat -p <backend_pid> on a second query execution.

Logging GUCs:
  This script deliberately does NOT set pgx_lower.log_enable, log_io, or
  enabled_categories.  The GUC defaults are all false/empty, giving a clean
  production-representative profile.  Do NOT add phase-timing GUCs here.
  If you need a logging-ON profile for debugging, use --log-enable explicitly.

Artifacts written to benchmark/profiling/perf-exec/<experiment>/:
  perf.data          — raw sampling data (perf record; gitignored)
  perf-stat.txt      — hardware counter summary (perf stat)
  perf-report.txt    — perf report --stdio, top symbols

Usage (inside the pgx-lower-dev container):
    python3 benchmark/profiling/perf-exec/run_perf_profile.py \\
        --query q01 --sf 1 --port 5432 \\
        --output benchmark/profiling/perf-exec/q01-sf1-nolog

Requires:
  - perf binary at /usr/local/bin/perf (installed in container by hand or
    baked into the Dockerfile; see benchmark/profiling/tooling.md)
  - kernel.perf_event_paranoid <= 1 on the host
    (docker-compose.yml has cap_add: [SYS_ADMIN, PERFMON] + seccomp:unconfined)
  - Running as root inside the container (to su to postgres for perf attach)
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psycopg2


QUERIES_DIR = Path(__file__).parent.parent.parent / "tpch" / "queries"

# ---------------------------------------------------------------------------
# JIT symbolization note
# ---------------------------------------------------------------------------
# MLIR ExecutionEngine sets enablePerfNotificationListener=True by default,
# which registers createPerfJITEventListener().  When perf record runs with
# -k 1 (CLOCK_MONOTONIC clockid), the LLVM JIT writes a jitdump file at
# /tmp/jit-<pid>.dump.  After recording, `perf inject --jit` merges that
# jitdump into perf.data, enabling JIT frame symbolization in perf report.
# Without -k 1, the jitdump is written but timestamps mismatch and perf
# inject cannot correlate samples to JIT frames.
# ---------------------------------------------------------------------------

PERF_STAT_EVENTS = (
    "cycles,instructions,branches,branch-misses,"
    "cache-references,cache-misses,"
    "stalled-cycles-frontend,stalled-cycles-backend"
)


def pg_connect(port: int) -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host="localhost", port=port, database="postgres", user="postgres"
    )


def reset_system_gucs(port: int) -> None:
    """
    Reset pgx_lower GUCs in postgresql.auto.conf via ALTER SYSTEM RESET.

    This prevents contamination from prior phase-timing runs that use
    ALTER SYSTEM SET pgx_lower.log_enable='on', which persists across
    sessions.  Session-level SET does NOT override ALTER SYSTEM in all
    cases — baking the RESET here guarantees a clean baseline.
    """
    conn = pg_connect(port)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("ALTER SYSTEM RESET pgx_lower.log_enable")
        cur.execute("ALTER SYSTEM RESET pgx_lower.log_io")
        cur.execute("ALTER SYSTEM RESET pgx_lower.enabled_categories")
        cur.execute("SELECT pg_reload_conf()")
    conn.close()
    print("  [reset_system_gucs] ALTER SYSTEM RESET log_enable/log_io/enabled_categories + pg_reload_conf()")


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


def get_backend_pid(conn: psycopg2.extensions.connection) -> int:
    """Return the postgres backend PID for this connection."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_backend_pid()")
        return cur.fetchone()[0]


def run_query_background(port: int, query_sql: str, log_enable: bool = False) -> tuple:
    """
    Execute the query in a background thread.

    Returns (thread, conn, result_holder).
    The thread loads pgx_lower, optionally enables logging GUCs, then executes
    the query.  The caller should attach perf AFTER the thread has started and
    the backend PID is known.

    IMPORTANT: log_enable defaults to False — this is the production-representative
    profile.  Do NOT set it True unless you specifically want a logging-ON profile.
    """
    result_holder = {"elapsed_ms": None, "error": None}

    conn = pg_connect(port)
    with conn.cursor() as cur:
        cur.execute("LOAD 'pgx_lower.so'")
        cur.execute("SET pgx_lower.enabled = on")
        # Explicitly ensure logging GUCs are off (belt-and-suspenders: the GUC
        # defaults are false, but set explicitly to guard against session residuals).
        if log_enable:
            cur.execute("SET pgx_lower.log_enable = on")
            cur.execute("SET pgx_lower.log_io = on")
            print("  WARNING: log_enable=on — this is a LOGGING-ON profile (contaminated for perf)")
        else:
            cur.execute("SET pgx_lower.log_enable = off")
            cur.execute("SET pgx_lower.log_io = off")

    backend_pid = get_backend_pid(conn)

    def _run():
        try:
            with conn.cursor() as cur:
                t0 = time.perf_counter()
                cur.execute(query_sql)
                cur.fetchall()
                result_holder["elapsed_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            result_holder["error"] = e
        finally:
            conn.close()

    t = threading.Thread(target=_run, daemon=True)
    return t, conn, backend_pid, result_holder


def run_query_sync(port: int, query_sql: str, log_enable: bool = False) -> float:
    """Execute the query synchronously and return elapsed ms."""
    conn = pg_connect(port)
    with conn.cursor() as cur:
        cur.execute("LOAD 'pgx_lower.so'")
        cur.execute("SET pgx_lower.enabled = on")
        cur.execute(f"SET pgx_lower.log_enable = {'on' if log_enable else 'off'}")
        cur.execute(f"SET pgx_lower.log_io = {'on' if log_enable else 'off'}")
        t0 = time.perf_counter()
        cur.execute(query_sql)
        cur.fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
    conn.close()
    return elapsed


def attach_perf_record(perf_bin: str, backend_pid: int, perf_data: str) -> subprocess.Popen:
    """
    Attach perf record to the backend PID.

    With paranoid=1, cross-user perf attach is blocked — the process running
    perf must be the same OS user as the profilee (postgres uid=1001).
    We use `su postgres -c "perf record -p ..."` to run perf as that user.

    perf_data MUST be an absolute path so it resolves correctly when perf runs
    in the postgres user's home directory (not the worktree CWD).

    Call-graph method: frame pointer (--call-graph fp) instead of DWARF.
    - DWARF produces accurate call graphs but 100–400MB files and multi-hour
      perf report processing times on Debug builds.
    - Frame pointer is less accurate in deeply optimized code but produces
      10–30MB files and fast (~30s) perf report processing.
    - For flat symbol profiling (which dominates occupies dominate?), fp is
      sufficient to identify hot functions; DWARF is not needed.

    Returns the Popen object so the caller can wait/interrupt it.
    """
    # Ensure the path is absolute so it works from any working directory
    abs_perf_data = str(Path(perf_data).resolve())
    cmd_str = (
        f"{perf_bin} record -g --call-graph fp "
        f"-k 1 "  # CLOCK_MONOTONIC: required for perf inject --jit JIT frame correlation
        f"-p {backend_pid} -o {abs_perf_data}"
    )
    # su postgres -c "..." — works when running as container root
    full_cmd = ["su", "postgres", "-c", cmd_str]
    print(f"  attaching: {' '.join(full_cmd)}")
    return subprocess.Popen(full_cmd)


def attach_perf_stat(perf_bin: str, backend_pid: int, duration_s: float) -> subprocess.CompletedProcess:
    """
    Run perf stat attached to the backend PID for a fixed duration.

    Used for the perf-stat pass (a second query execution).
    """
    cmd_str = (
        f"{perf_bin} stat -e {PERF_STAT_EVENTS} "
        f"-p {backend_pid} -- sleep {duration_s:.1f}"
    )
    return subprocess.run(["su", "postgres", "-c", cmd_str],
                          capture_output=True, text=True)


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
    parser.add_argument(
        "--log-enable", action="store_true",
        help="Enable pgx_lower.log_enable and log_io (produces a LOGGING-ON profile, "
             "useful for isolating logging overhead but NOT production-representative)",
    )
    args = parser.parse_args()

    check_perf(args.perf_bin)

    # Reset system-level GUCs before any query runs.
    # This prevents contamination from prior ALTER SYSTEM SET calls (e.g., from
    # phase-timing runs).  Must run before ensure_tpch_loaded and warm-up.
    print("\n=== Resetting system GUCs (ALTER SYSTEM RESET) ===")
    try:
        reset_system_gucs(args.port)
    except Exception as e:
        print(f"  WARNING: could not reset GUCs: {e} — continuing (session SET will still override)")

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

    # Ensure the output directory AND all its parents are executable by the
    # postgres user (uid=1001) so that `su postgres -c "perf record -o ..."` can
    # write perf.data.  The container runs as root, so parent dirs are often
    # mode 700 (drwx------).  We need at least o+x on each ancestor.
    p = out_dir.resolve()
    while str(p) != p.root:
        st = p.stat()
        if not (st.st_mode & 0o001):  # other-execute bit
            p.chmod(st.st_mode | 0o001)
        p = p.parent

    # Make the output dir itself world-writable so postgres can create files in it
    out_dir.chmod(0o777)

    perf_data = str(out_dir / "perf.data")
    perf_stat_txt = str(out_dir / "perf-stat.txt")
    perf_report_txt = str(out_dir / "perf-report.txt")

    ensure_tpch_loaded(args.port, args.sf)

    query_sql = query_sql_path.read_text()

    if not args.skip_warmup:
        print(f"\n=== Warm-up run ({args.query} @ SF={args.sf}) ===")
        elapsed = run_query_sync(args.port, query_sql, log_enable=args.log_enable)
        print(f"  warm-up: {elapsed:.0f} ms")

    # ── perf record (attach to backend) ──────────────────────────────────────
    print(f"\n=== perf record ({args.query} @ SF={args.sf}) ===")
    print(f"  method: attach to postgres backend PID (not psql client)")
    print(f"  logging: {'ON (--log-enable)' if args.log_enable else 'OFF (production-representative)'}")
    print(f"  output: {perf_data}")

    # Start query on background thread; get backend PID before query starts
    thread, conn_unused, backend_pid, result = run_query_background(
        args.port, query_sql, log_enable=args.log_enable
    )
    print(f"  backend PID: {backend_pid}")

    # Start perf BEFORE starting the query thread so we don't miss early samples
    perf_proc = attach_perf_record(args.perf_bin, backend_pid, perf_data)

    # Brief pause to let perf initialize before the query starts executing
    time.sleep(0.2)

    # Start the query
    t0 = time.perf_counter()
    thread.start()

    # Wait for query to complete
    thread.join(timeout=300)
    elapsed_total = time.perf_counter() - t0

    if thread.is_alive():
        sys.exit("ERROR: query thread timed out after 300 s")
    if result["error"]:
        perf_proc.send_signal(signal.SIGINT)
        perf_proc.wait()
        sys.exit(f"ERROR: query failed: {result['error']}")

    query_ms = result["elapsed_ms"]
    print(f"  query elapsed: {query_ms:.0f} ms")

    # Stop perf — send SIGINT so it flushes perf.data cleanly
    try:
        perf_proc.send_signal(signal.SIGINT)
    except ProcessLookupError:
        pass  # already exited
    perf_proc.wait()
    print(f"  total elapsed (query + perf overhead): {elapsed_total * 1000:.0f} ms, perf exit: {perf_proc.returncode}")

    size_bytes = Path(perf_data).stat().st_size if Path(perf_data).exists() else 0
    print(f"  perf.data size: {size_bytes:,} bytes")

    if size_bytes == 0:
        print("WARNING: perf.data is empty — perf attach may have failed.")
        print("  Check that su postgres works and paranoid=1 is set on the host.")

    # ── perf inject --jit (merge jitdump for JIT frame symbolization) ─────────
    # MLIR ExecutionEngine registers createPerfJITEventListener() by default,
    # which writes /tmp/jit-<pid>.dump when perf record runs with -k 1.
    # perf inject --jit merges that jitdump into perf.data so JIT frames
    # appear with function names instead of raw addresses in perf report.
    jit_data = str(out_dir / "perf-jit.data")
    if size_bytes > 0:
        print(f"\n=== perf inject --jit (JIT frame symbolization) ===")
        abs_jit_data = str(Path(jit_data).resolve())
        abs_perf_data_inject = str(Path(perf_data).resolve())
        inject_cmd_str = (
            f"{args.perf_bin} inject --jit "
            f"-i {abs_perf_data_inject} "
            f"-o {abs_jit_data}"
        )
        # Run as postgres since perf.data is postgres-owned
        inject_result = subprocess.run(
            ["su", "postgres", "-c", inject_cmd_str],
            capture_output=True, text=True,
        )
        print(f"  exit: {inject_result.returncode}")
        if inject_result.returncode == 0:
            jit_size = Path(abs_jit_data).stat().st_size if Path(abs_jit_data).exists() else 0
            print(f"  perf-jit.data written ({jit_size:,} bytes) — JIT frames symbolized")
            # Use jit-injected data for the report
            perf_data_for_report = abs_jit_data
        else:
            print(f"  WARNING: perf inject --jit failed: {inject_result.stderr[:400]}")
            print(f"  Falling back to raw perf.data for report (JIT frames as raw addresses)")
            perf_data_for_report = str(Path(perf_data).resolve())
    else:
        perf_data_for_report = str(Path(perf_data).resolve())

    # ── perf stat (second query execution, attach via -p + sleep) ────────────
    print(f"\n=== perf stat ({args.query} @ SF={args.sf}) ===")
    print(f"  output: {perf_stat_txt}")

    # Run the query on a background thread; attach perf stat via -p + sleep
    thread2, conn2_unused, backend_pid2, result2 = run_query_background(
        args.port, query_sql, log_enable=args.log_enable
    )
    print(f"  backend PID: {backend_pid2}")

    # Estimate stat duration: add 20% margin over measured query time
    stat_duration_s = max(10.0, (query_ms / 1000.0) * 1.3)
    print(f"  stat duration: {stat_duration_s:.1f} s")

    thread2.start()

    # Run perf stat as postgres user, attached to the backend PID.
    # Use `perf stat -a -p <pid> -- sleep N` to measure the backend for the
    # query duration.  The sleep command anchors the stat window; the backend
    # pid is the target.
    stat_cmd_str = (
        f"{args.perf_bin} stat -e {PERF_STAT_EVENTS} "
        f"-p {backend_pid2} -- /bin/sleep {stat_duration_s:.1f}"
    )
    stat_result = subprocess.run(
        ["su", "postgres", "-c", stat_cmd_str],
        capture_output=True, text=True,
    )
    thread2.join(timeout=60)

    with open(perf_stat_txt, "w") as stat_out:
        import datetime
        stat_out.write(
            f"# perf stat: {args.query} @ SF={args.sf}\n"
            f"# method: perf stat -p <backend_pid> as postgres user\n"
            f"# logging: {'ON (--log-enable flag)' if args.log_enable else 'OFF (default, production-representative)'}\n"
            f"# date: {datetime.date.today()}, paranoid=1, container caps: SYS_ADMIN+PERFMON, seccomp:unconfined\n"
            f"# perf version 6.8.1\n"
            f"# command: su postgres -c \"{stat_cmd_str}\"\n\n"
        )
        # perf stat writes to stderr
        combined = stat_result.stdout + stat_result.stderr
        stat_out.write(combined)

    print(f"  exit: {stat_result.returncode}")
    if Path(perf_stat_txt).stat().st_size > 0:
        print(f"  perf-stat.txt written ({Path(perf_stat_txt).stat().st_size:,} bytes)")

    # ── perf report ──────────────────────────────────────────────────────────
    if Path(perf_data).exists() and size_bytes > 0:
        print(f"\n=== perf report ({args.query} @ SF={args.sf}) ===")
        # Use jit-injected data if available (for JIT frame symbolization),
        # else fall back to raw perf.data.
        report_input = perf_data_for_report if 'perf_data_for_report' in dir() else str(Path(perf_data).resolve())
        abs_report_txt = str(Path(perf_report_txt).resolve())
        print(f"  input: {report_input}")
        # perf.data is owned by postgres (written by su postgres perf record).
        # perf report must run as the same user OR with -f (force).
        # Use su postgres to avoid ownership errors; pipe stdout through
        # the root shell's redirect so perf-report.txt stays root-owned.
        # Suppress addr2line stderr noise (very verbose on Debug builds with
        # large DWARF — 2>/dev/null sends those errors to null).
        report_cmd_str = (
            f"{args.perf_bin} report --stdio --no-children "
            f"-i {report_input} 2>/dev/null"
        )
        with open(perf_report_txt, "w") as report_out:
            report_result = subprocess.run(
                ["su", "postgres", "-c", report_cmd_str],
                stdout=report_out,
                stderr=subprocess.DEVNULL,
            )
        print(f"  exit: {report_result.returncode}")
        size_report = Path(perf_report_txt).stat().st_size if Path(perf_report_txt).exists() else 0
        print(f"  perf-report.txt written ({size_report:,} bytes)")

        # Show top symbols + check for logging contamination + FFI symbols
        print("\n=== Top symbols ===")
        log_symbols = ["should_log", "_Rb_tree", "ScopeLogger", "log::log"]
        ffi_symbols = ["extract_field", "get_int32_field_mlir", "get_int64_field_mlir",
                       "get_date_field_mlir", "get_decimal_field_mlir", "get_float8_field_mlir"]
        with open(perf_report_txt) as f:
            content = f.read()

        # Print top 30 non-comment lines
        lines = [l for l in content.splitlines() if not l.startswith("#") and l.strip()]
        for line in lines[:30]:
            print(" ", line)

        print("\n=== Logging contamination check ===")
        # Only check flat-profile lines (start with whitespace + digits + %)
        # NOT call-graph child lines (which start with | or space).
        flat_log_lines = [
            l for l in content.splitlines()
            if l.strip() and l.strip()[0].isdigit() and "%" in l
            and any(s in l for s in log_symbols)
        ]
        if flat_log_lines:
            log_pct = 0.0
            for l in flat_log_lines:
                try:
                    log_pct += float(l.strip().split("%")[0])
                except ValueError:
                    pass
            print(f"  Logging symbols in flat profile: {len(flat_log_lines)} entries")
            print(f"  Estimated flat logging overhead: ~{log_pct:.1f}%")
            if log_pct > 15.0:
                print(f"  ALERT: >15% logging in flat profile — likely GUC-ON contamination.")
                print(f"  Check: ALTER SYSTEM RESET pgx_lower.log_enable (and reload conf).")
                print(f"  Also check: SELECT name, setting FROM pg_settings WHERE name LIKE 'pgx_lower%';")
            elif log_pct > 5.0:
                print(f"  NOTE: 5-15% logging overhead — likely real PGX_IO per-tuple cost")
                print(f"  (should_log() + std::optional<ScopeLogger> ctor/dtor per PGX_IO call)")
                print(f"  + possible frame-pointer misattribution from JITed code.")
            else:
                print(f"  CLEAN: <5% logging overhead — no contamination detected.")
        else:
            print(f"  CLEAN: no logging symbols in flat profile — contamination absent")

        print("\n=== FFI symbol check ===")
        found_ffi = [sym for sym in ffi_symbols if sym in content]
        if found_ffi:
            print(f"  FOUND: {found_ffi}")
        else:
            print("  NOT FOUND: no extract_field / get_*_field_mlir in perf report")
            print("  (This may mean the workload is not FFI-bound, or symbols need --demangle.)")

    print(f"\n=== Artifacts written to: {out_dir} ===")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
