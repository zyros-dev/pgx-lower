#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import select
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import lz4.frame
import psutil
import psycopg2

from metrics_collector import collect_query_metrics
from fxt_to_flamegraph import fxt_to_flamegraph_json


def get_cpu_vendor() -> str:
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('vendor_id'):
                    vendor = line.split(':')[1].strip()
                    if 'Intel' in vendor:
                        return 'intel'
                    elif 'AMD' in vendor:
                        return 'amd'
    except Exception:
        pass
    return 'unknown'


def check_magic_trace_available() -> bool:
    try:
        result = subprocess.run(['which', 'magic-trace'], capture_output=True, check=True)
        return result.returncode == 0
    except:
        return False


SCHEMA_SQL = """
             CREATE TABLE IF NOT EXISTS runs
             (
                 run_id           TEXT PRIMARY KEY,
                 run_timestamp    TEXT    NOT NULL,
                 scale_factor     REAL    NOT NULL,
                 iterations       INTEGER NOT NULL,
                 postgres_version TEXT,
                 pgx_version      TEXT,
                 hostname         TEXT,
                 run_args         TEXT,
                 container        TEXT
             );

             CREATE TABLE IF NOT EXISTS queries
             (
                 query_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                 run_id             TEXT    NOT NULL,
                 query_name         TEXT    NOT NULL,
                 iteration          INTEGER NOT NULL,
                 pgx_enabled        BOOLEAN NOT NULL,
                 execution_metadata TEXT,
                 metrics_json       TEXT,
                 postgres_metrics   TEXT,
                 timeseries_metrics TEXT,
                 result_validation  TEXT,
                 FOREIGN KEY (run_id) REFERENCES runs (run_id)
             );

             CREATE INDEX IF NOT EXISTS idx_run_query ON queries (run_id, query_name);
             CREATE INDEX IF NOT EXISTS idx_pgx ON queries (pgx_enabled);

             CREATE TABLE IF NOT EXISTS profiling
             (
                 profile_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 query_id            INTEGER NOT NULL,
                 cpu_data_lz4        BLOB,
                 heap_data_lz4       BLOB,
                 cpu_flamegraph_lz4  BLOB,
                 heap_flamegraph_lz4 BLOB,
                 profile_metadata    TEXT,
                 FOREIGN KEY (query_id) REFERENCES queries (query_id)
             );

             CREATE TABLE IF NOT EXISTS aggregate_benchmarks
             (
                 id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                 pgx_version        TEXT    NOT NULL,
                 postgres_version   TEXT    NOT NULL,
                 scale_factor       REAL    NOT NULL,
                 pgx_enabled        BOOLEAN NOT NULL,
                 run_timestamp      TEXT    NOT NULL,
                 execution_metadata TEXT,
                 metrics_json       TEXT,
                 postgres_metrics   TEXT,
                 UNIQUE (pgx_version, postgres_version, scale_factor, pgx_enabled)
             );

             CREATE TABLE IF NOT EXISTS aggregate_profiles
             (
                 id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                 pgx_version         TEXT    NOT NULL,
                 postgres_version    TEXT    NOT NULL,
                 scale_factor        REAL    NOT NULL,
                 pgx_enabled         BOOLEAN NOT NULL,
                 run_timestamp       TEXT    NOT NULL,
                 cpu_data_lz4        BLOB,
                 heap_data_lz4       BLOB,
                 cpu_flamegraph_lz4  BLOB,
                 heap_flamegraph_lz4 BLOB,
                 profile_metadata    TEXT,
                 UNIQUE (pgx_version, postgres_version, scale_factor, pgx_enabled)
             ); \
             """


def generate_tpch_data(scale_factor):
    benchmark_dir = Path(__file__).parent.parent
    script = benchmark_dir / 'generate_tpch_data.py'
    output_file = benchmark_dir / 'tpch_init.sql'

    print(f"Generating TPC-H data at SF={scale_factor}...")
    try:
        result = subprocess.run(
            ['python3', str(script), str(scale_factor), '-o', str(output_file)],
            cwd=benchmark_dir,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"ERROR generating data: {result.stderr}")
            return False

        if not output_file.exists():
            print(f"ERROR: {output_file} not created")
            return False

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"Generated {file_size_mb:.1f} MB SQL file")
        return True

    except subprocess.TimeoutExpired:
        print("ERROR: Data generation timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def load_tpch_database(conn):
    sql_file = Path(__file__).parent.parent / 'tpch_init.sql'

    if not sql_file.exists():
        print(f"ERROR: {sql_file} not found")
        return False

    print(f"Loading TPC-H database...")
    load_start = time.time()
    try:
        with open(sql_file, 'r') as f:
            sql = f.read()

        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        load_elapsed = time.time() - load_start

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customer")
            count = cur.fetchone()[0]
            print(f"Loaded {count} customers in {load_elapsed:.2f}s")

        return True

    except Exception as e:
        print(f"ERROR loading database: {e}")
        conn.rollback()
        return False


def get_postgres_version(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        return cur.fetchone()[0]


def get_pgx_version():
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except:
        return 'unknown'


def normalize_value(val):
    if val is None:
        return 'NULL'

    try:
        f = float(val)
        return f"{f:.4f}"
    except (ValueError, TypeError):
        pass

    return str(val).strip()


def run_query_with_metrics(conn, query_file, pgx_enabled, iteration, profile_enabled=False,
                           output_dir=None, db_conn=None, cpu_vendor=None, magic_trace_available=False, scale_factor=0.01):
    query_name = query_file.stem

    with open(query_file, 'r') as f:
        query_sql = f.read()

    process = psutil.Process()
    io_start = process.io_counters()
    cpu_start = process.cpu_times()

    result = {
        'query_name': query_name,
        'iteration': iteration,
        'pgx_enabled': pgx_enabled,
        'status': 'SUCCESS',
        'error_message': None,
        'duration_ms': 0,
        'row_count': 0
    }

    detailed_metrics = {}
    timeseries = []
    sampling_active = True
    magic_trace_proc = None
    magic_trace_file = None

    def sample_metrics():
        start = time.time()
        while sampling_active:
            try:
                elapsed_sec = time.time() - start
                elapsed_ms = elapsed_sec * 1000

                mem = process.memory_info()
                io = process.io_counters()
                cpu = process.cpu_times()

                sample = {
                    't_ms': round(elapsed_ms, 2),
                    'mem_mb': round(mem.rss / (1024 * 1024), 2),
                    'io_r_mb': round(io.read_bytes / (1024 * 1024), 2),
                    'io_w_mb': round(io.write_bytes / (1024 * 1024), 2),
                    'cpu_u': round(cpu.user, 3),
                    'cpu_s': round(cpu.system, 3)
                }
                timeseries.append(sample)

                interval_ms = min(5 + (elapsed_sec * 5), 30000)
                time.sleep(interval_ms / 1000)
            except:
                break

    sampler_thread = threading.Thread(target=sample_metrics, daemon=True)
    sampler_thread.start()

    try:
        with conn.cursor() as cur:
            backend_pid = None
            if profile_enabled and magic_trace_available:
                cur.execute("SELECT pg_backend_pid()")
                backend_pid = cur.fetchone()[0]

                pgx_str = 'on' if pgx_enabled else 'off'
                trace_dir = output_dir / 'profiles_cpu_latest'
                magic_trace_file = str(trace_dir / f'{query_name}_pgx_{pgx_str}.fxt')
                magic_trace_log = str(trace_dir / f'{query_name}_pgx_{pgx_str}.log')

                perf_data_dir = trace_dir / 'perf_data'
                perf_data_dir.mkdir(exist_ok=True)
                working_dir = str(perf_data_dir / f'{query_name}_pgx_{pgx_str}')

                magic_trace_cmd = ['sudo', 'magic-trace', 'attach', '-pid', str(backend_pid),
                                   '-output', magic_trace_file, '-working-directory', working_dir]

                if cpu_vendor == 'amd':
                    magic_trace_cmd.extend(['-sampling', '-timer-resolution', 'High', '-callgraph-mode', 'fp'])
                else:
                    # Intel: Use LBR sampling with frequency adjusted by scale factor
                    # LBR (Last Branch Record) provides hardware stack traces without frame pointers/DWARF
                    # -timer-resolution controls sampling frequency: Low=1000/s, Normal=10000/s, High=max
                    if scale_factor <= 0.01:
                        # Small scale factor: High frequency for detailed profiling
                        # Fast queries (~30-300ms): ~20-25 samples/ms, ~0.2-4MB files
                        magic_trace_cmd.extend(['-sampling', '-callgraph-mode', 'lbr',
                                              '-timer-resolution', 'High', '-full-execution'])
                    else:
                        magic_trace_cmd.extend(['-sampling', '-callgraph-mode', 'lbr',
                                              '-timer-resolution', 'Low', '-full-execution'])

                magic_trace_log_file = open(magic_trace_log, 'w')
                magic_trace_proc = subprocess.Popen(
                    magic_trace_cmd,
                    stdout=magic_trace_log_file,
                    stderr=subprocess.STDOUT
                )

                attach_timeout = 5.0
                start_wait = time.time()
                attached = False
                while time.time() - start_wait < attach_timeout:
                    magic_trace_log_file.flush()
                    try:
                        with open(magic_trace_log, 'r') as check_log:
                            log_content = check_log.read()
                            if 'Attached' in log_content:
                                attached = True
                                break
                    except:
                        pass
                    time.sleep(0.05)

                if not attached:
                    print(f"  Warning: magic-trace did not attach within {attach_timeout}s", end=' ')
                else:
                    time.sleep(0.2)

            start_time = time.time()
            cur.execute(query_sql)

            # Hash first 5000 rows without loading all into memory
            hasher = hashlib.sha256()
            row_count = 0
            for row in cur:
                row_count += 1
                if row_count <= 5000:
                    # Normalize each value in the row
                    normalized = [normalize_value(val) for val in row]
                    row_str = '|'.join(normalized) + '\n'
                    hasher.update(row_str.encode('utf-8'))

            duration_ms = (time.time() - start_time) * 1000

            result['duration_ms'] = duration_ms
            result['row_count'] = row_count
            result['result_hash'] = hasher.hexdigest()

            # Collect PostgreSQL metrics (EXPLAIN ANALYZE, buffers, memory contexts)
            try:
                result['postgres_metrics'] = collect_query_metrics(cur, query_sql)
            except Exception as e:
                print(f"  Warning: Failed to collect PG metrics: {e}")
                result['postgres_metrics'] = None

            # Stop magic-trace after query execution completes
            if magic_trace_proc is not None:
                try:
                    # Send SIGINT to stop recording (for full-execution mode)
                    print(f"[query done, decoding trace...]", end=' ', flush=True)
                    magic_trace_proc.send_signal(signal.SIGINT)
                    magic_trace_proc.wait(timeout=180)
                    print(f"[decode done]", end=' ', flush=True)

                    username = os.environ.get('USER') or os.environ.get('LOGNAME')
                    if username and os.path.exists(magic_trace_file):
                        subprocess.run(['sudo', 'chown', username, magic_trace_file],
                                       stderr=subprocess.DEVNULL, check=False)
                        subprocess.run(['sudo', 'chmod', '644', magic_trace_file],
                                       stderr=subprocess.DEVNULL, check=False)
                except subprocess.TimeoutExpired:
                    print(f"  Warning: magic-trace decode timeout (180s), killing", end=' ')
                    try:
                        magic_trace_proc.kill()
                    except:
                        pass
                except Exception as e:
                    print(f"  Warning: magic-trace error: {e}", end=' ')
                    try:
                        magic_trace_proc.kill()
                    except:
                        pass
                finally:
                    if 'magic_trace_log_file' in locals():
                        magic_trace_log_file.close()

        conn.commit()

    except Exception as e:
        result['status'] = 'ERROR'
        result['error_message'] = str(e)
        conn.rollback()

        if magic_trace_proc is not None:
            try:
                magic_trace_proc.kill()
            except:
                pass

    finally:
        sampling_active = False
        sampler_thread.join(timeout=0.1)

    io_end = process.io_counters()
    cpu_end = process.cpu_times()
    mem_end = process.memory_info()

    detailed_metrics['io_read_mb'] = (io_end.read_bytes - io_start.read_bytes) / (1024 * 1024)
    detailed_metrics['io_write_mb'] = (io_end.write_bytes - io_start.write_bytes) / (1024 * 1024)
    detailed_metrics['cpu_user_sec'] = cpu_end.user - cpu_start.user
    detailed_metrics['cpu_system_sec'] = cpu_end.system - cpu_start.system
    detailed_metrics['memory_peak_mb'] = mem_end.rss / (1024 * 1024)

    try:
        ctx_switches = process.num_ctx_switches()
        detailed_metrics['context_switches_involuntary'] = ctx_switches.involuntary
        detailed_metrics['context_switches_voluntary'] = ctx_switches.voluntary
    except:
        detailed_metrics['context_switches_involuntary'] = 0
        detailed_metrics['context_switches_voluntary'] = 0

    cpu_total = detailed_metrics['cpu_user_sec'] + detailed_metrics['cpu_system_sec']
    detailed_metrics['cpu_percent'] = (cpu_total / (result['duration_ms'] / 1000)) * 100 if result[
                                                                                                'duration_ms'] > 0 else 0

    result['metrics_json'] = json.dumps(detailed_metrics)
    result['timeseries_metrics'] = json.dumps(timeseries)
    result['magic_trace_file'] = magic_trace_file

    return result


def insert_metrics(db_conn, run_id, result, run_timestamp, pgx_version, postgres_version, scale_factor):
    cursor = db_conn.cursor()

    execution_metadata = json.dumps({
        'status': result['status'],
        'duration_ms': result['duration_ms'],
        'row_count': result['row_count'],
        'error_message': result['error_message']
    })

    result_validation = json.dumps({
        'hash': result.get('result_hash'),
        'valid': result.get('hash_valid', None)
    })

    cursor.execute("""
                   INSERT INTO queries (run_id, query_name, iteration, pgx_enabled,
                                        execution_metadata, metrics_json, postgres_metrics,
                                        timeseries_metrics, result_validation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   """, (
                       run_id,
                       result['query_name'],
                       result['iteration'],
                       result['pgx_enabled'],
                       execution_metadata,
                       result['metrics_json'],
                       result.get('postgres_metrics'),
                       result['timeseries_metrics'],
                       result_validation
                   ))
    db_conn.commit()
    return cursor.lastrowid


def timeout_handler():
    print("Starting timeout")
    time.sleep(300)
    os.kill(os.getpid(), signal.SIGKILL)


def setup_profiling_dirs(output_dir, run_timestamp, scale_factor):
    ts_clean = run_timestamp.replace(':', '').replace('-', '').replace('T', '_').split('.')[0]
    dirpath = output_dir / 'traces' / f'sf_{scale_factor}_{ts_clean}'
    dirpath.mkdir(parents=True, exist_ok=True)

    latest_link = output_dir / 'profiles_cpu_latest'
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        shutil.rmtree(latest_link)
    latest_link.symlink_to(dirpath.relative_to(output_dir))

    return dirpath


def init_databases(sf, run_id, output_dir, run_timestamp, pg_port=5432, run_args=None, container=None):
    db_file = output_dir / 'benchmark.db'
    db_conn = sqlite3.connect(db_file)
    db_conn.executescript(SCHEMA_SQL)

    # Kill all existing connections with OS-level kill and drop TPC-H tables
    cleanup_conn = psycopg2.connect(host='localhost', port=pg_port, database='postgres', user='postgres')
    cleanup_conn.autocommit = True
    try:
        with cleanup_conn.cursor() as cur:
            # Get all client backend PIDs except our own
            cur.execute("""
                        SELECT pid
                        FROM pg_stat_activity
                        WHERE pid != pg_backend_pid()
                          AND backend_type = 'client backend'
                        """)
            pids = [row[0] for row in cur.fetchall()]

            # Force kill at OS level
            if pids:
                pid_list = ' '.join(str(p) for p in pids)
                result = subprocess.run(
                    ['docker', 'compose', '-f', 'docker/docker-compose.yml', 'exec', 'benchmark', 'bash', '-c',
                     f'kill -9 {pid_list}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                print(f"Cleaned up {len(pids)} stale connections")
                time.sleep(0.5)

            # Drop all TPC-H tables to ensure clean state
            cur.execute("""
                        DROP TABLE IF EXISTS lineitem CASCADE;
                        DROP TABLE IF EXISTS orders CASCADE;
                        DROP TABLE IF EXISTS customer CASCADE;
                        DROP TABLE IF EXISTS partsupp CASCADE;
                        DROP TABLE IF EXISTS supplier CASCADE;
                        DROP TABLE IF EXISTS part CASCADE;
                        DROP TABLE IF EXISTS nation CASCADE;
                        DROP TABLE IF EXISTS region CASCADE;
                        """)
    except Exception as e:
        print(f"Warning: Failed to clean up: {e}")
    finally:
        cleanup_conn.close()

    pg_conn = psycopg2.connect(host='localhost', port=pg_port, database='postgres', user='postgres')
    pg_conn.autocommit = False

    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT INTO runs (run_id, run_timestamp, scale_factor, iterations, postgres_version, pgx_version, hostname, run_args, container) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, run_timestamp, sf, 1, get_postgres_version(pg_conn), get_pgx_version(),
         subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), run_args, container))
    db_conn.commit()

    return db_conn, pg_conn, db_file


def validate_results(db_conn, run_id):
    print(f"\nValidating results...")
    cursor = db_conn.cursor()

    cursor.execute("""
                   SELECT query_name,
                          MAX(CASE
                                  WHEN pgx_enabled = 0
                                      THEN json_extract(result_validation, '$.hash') END)                   AS postgres_hash,
                          MAX(CASE WHEN pgx_enabled = 1 THEN json_extract(result_validation, '$.hash') END) AS pgx_hash
                   FROM queries
                   WHERE run_id = ?
                     AND json_extract(execution_metadata, '$.status') = 'SUCCESS'
                   GROUP BY query_name
                   """, (run_id,))

    mismatches = []
    for row in cursor.fetchall():
        query_name, postgres_hash, pgx_hash = row
        is_valid = (postgres_hash == pgx_hash) if (postgres_hash and pgx_hash) else None

        # Update result_validation JSON with valid field
        cursor.execute("""
                       UPDATE queries
                       SET result_validation = json_set(result_validation, '$.valid', ?)
                       WHERE run_id = ?
                         AND query_name = ?
                       """, (is_valid, run_id, query_name))

        if is_valid == False:
            mismatches.append(query_name)

    db_conn.commit()

    if mismatches:
        print(f"\nHash mismatches detected in: {', '.join(mismatches)}")
    else:
        print(f"\nAll query results validated!")


def run_benchmark_queries(pg_conn, db_conn, run_id, script_dir, profile_enabled, output_dir, run_timestamp,
                          pgx_version, postgres_version, scale_factor, db_file, pg_port=5432, query_filter=None):
    query_files = sorted((script_dir / 'queries').glob('q*.sql'))

    if query_filter:
        query_files = [qf for qf in query_files if qf.stem == query_filter]
        if not query_files:
            print(f"Error: Query '{query_filter}' not found")
            sys.exit(1)

    cpu_vendor = get_cpu_vendor()
    magic_trace_available = check_magic_trace_available()

    if profile_enabled:
        if magic_trace_available:
            mode_str = "Intel PT" if cpu_vendor == "intel" else "sampling" if cpu_vendor == "amd" else "unknown"
            print(f"magic-trace enabled ({cpu_vendor} CPU, {mode_str} mode)")
        else:
            print("Warning: magic-trace not found, profiling disabled")
            profile_enabled = False

    total = len(query_files) * 2
    current = 0

    print(f"\n{total} queries...")

    for pgx_enabled in [False, True]:
        if pgx_enabled and pg_conn:
            pg_conn.close()
            pg_conn = psycopg2.connect(host='localhost', port=pg_port, database='postgres', user='postgres')
            pg_conn.autocommit = False

            with pg_conn.cursor() as cur:
                cur.execute("LOAD 'pgx_lower.so'")
                cur.execute("SET pgx_lower.enabled = true")
            pg_conn.commit()

        pgx_mode = 'ON ' if pgx_enabled else 'OFF'
        print(f"\n{'pgx-lower' if pgx_enabled else 'PostgreSQL'} (pgx-lower {pgx_mode}):")

        for qf in query_files:
            current += 1
            print(f"[{current}/{total}] {qf.stem}...", end=' ', flush=True)

            metrics = run_query_with_metrics(
                pg_conn, qf, pgx_enabled, 1,
                profile_enabled=profile_enabled,
                output_dir=output_dir,
                db_conn=db_conn,
                cpu_vendor=cpu_vendor,
                magic_trace_available=magic_trace_available,
                scale_factor=scale_factor
            )
            query_id = insert_metrics(db_conn, run_id, metrics, run_timestamp, pgx_version, postgres_version,
                                      scale_factor)

            if profile_enabled and metrics['status'] == 'SUCCESS' and metrics.get('magic_trace_file'):
                trace_file = Path(metrics['magic_trace_file'])

                if not trace_file.exists():
                    print("[fxt file missing]", end=' ')
                else:
                    trace_size_kb = trace_file.stat().st_size / 1024

                    flamegraph_blob = fxt_to_flamegraph_json(trace_file)
                    flamegraph_size_kb = len(flamegraph_blob) / 1024 if flamegraph_blob else None

                    with open(trace_file, 'rb') as f:
                        fxt_data = f.read()
                    fxt_compressed = lz4.frame.compress(fxt_data, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)
                    fxt_compressed_kb = len(fxt_compressed) / 1024

                    profile_metadata = json.dumps({
                        'magic_trace': {
                            'file': str(trace_file),
                            'raw_kb': trace_size_kb,
                            'compressed_kb': fxt_compressed_kb,
                            'compression_ratio': trace_size_kb / fxt_compressed_kb if fxt_compressed_kb > 0 else None,
                            'cpu_vendor': cpu_vendor,
                            'mode': 'intel_pt' if cpu_vendor == 'intel' else 'sampling' if cpu_vendor == 'amd' else 'unknown'
                        },
                        'flamegraph': {
                            'size_kb': flamegraph_size_kb,
                            'available': flamegraph_blob is not None
                        }
                    })

                    cursor = db_conn.cursor()
                    cursor.execute("""
                                   INSERT INTO profiling (query_id, cpu_data_lz4, cpu_flamegraph_lz4, profile_metadata)
                                   VALUES (?, ?, ?, ?)
                                   """, (query_id, fxt_compressed, flamegraph_blob, profile_metadata))
                    db_conn.commit()

            if metrics['status'] == 'SUCCESS':
                hash_short = metrics['result_hash'][:8]
                print(f"{metrics['duration_ms']:.1f}ms [{hash_short}]")
            else:
                print(f"ERROR: {metrics['error_message']}")

    print("\nComputing aggregate statistics...")
    from aggregate import aggregate_latest_run
    aggregate_latest_run(db_file, run_timestamp)


def main():
    threading.Thread(target=timeout_handler, daemon=True).start()

    parser = argparse.ArgumentParser(description='Run TPC-H benchmarks for pgx-lower')
    parser.add_argument('scale_factor', type=float, nargs='?', default=0.01,
                        help='TPC-H scale factor (default: 0.01)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling with magic-trace (auto-detects Intel PT or AMD sampling mode)')
    parser.add_argument('--heap', action='store_true',
                        help='Enable heap profiling (reserved for future use)')
    parser.add_argument('--query', '-q', type=str,
                        help='Run only a specific query (e.g., q01)')
    parser.add_argument('--port', '-p', type=int, default=5432,
                        help='PostgreSQL port (default: 5432)')
    parser.add_argument('--container', type=str, default='unknown',
                        help='Container name (for tracking which container this run came from)')
    args = parser.parse_args()

    sf = args.scale_factor
    profile_enabled = args.profile
    query_filter = args.query
    container_name = args.container
    pg_port = args.port

    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().isoformat()
    run_id = f"pgx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_sf{int(sf * 100):03d}"

    profiles_dir = None
    if profile_enabled:
        profiles_dir = setup_profiling_dirs(output_dir, run_timestamp, sf)
    print(f"{run_id} (SF={sf})")

    run_args_parts = [f"SF={sf}"]
    if profile_enabled:
        run_args_parts.append("--profile")
    if query_filter:
        run_args_parts.append(f"--query={query_filter}")
    run_args = " ".join(run_args_parts)

    db_conn, pg_conn, db_file = init_databases(sf, run_id, output_dir, run_timestamp, pg_port, run_args, container_name)

    postgres_version = get_postgres_version(pg_conn)
    pgx_version = get_pgx_version()

    if not generate_tpch_data(sf):
        sys.exit(1)

    if not load_tpch_database(pg_conn):
        sys.exit(1)

    run_benchmark_queries(pg_conn, db_conn, run_id, script_dir, profile_enabled, output_dir, run_timestamp,
                          pgx_version, postgres_version, sf, db_file, pg_port, query_filter)

    pg_conn.close()
    validate_results(db_conn, run_id)
    db_conn.close()

    print(f"\nDone: {db_file}")


if __name__ == '__main__':
    main()
