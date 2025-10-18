#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
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

SCHEMA_SQL = """
             CREATE TABLE IF NOT EXISTS runs
             (
                 run_id           TEXT PRIMARY KEY,
                 timestamp        INTEGER NOT NULL,
                 scale_factor     REAL    NOT NULL,
                 iterations       INTEGER NOT NULL,
                 postgres_version TEXT,
                 pgx_version      TEXT,
                 hostname         TEXT
             );

             CREATE TABLE IF NOT EXISTS queries
             (
                 query_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                 run_id             TEXT    NOT NULL,
                 query_name         TEXT    NOT NULL,
                 iteration          INTEGER NOT NULL,
                 jit_enabled        BOOLEAN NOT NULL,
                 status             TEXT    NOT NULL,
                 duration_ms        REAL,
                 row_count          INTEGER,
                 error_message      TEXT,
                 metrics_json       TEXT,
                 postgres_metrics   TEXT,
                 timeseries_metrics TEXT,
                 result_hash        TEXT,
                 hash_valid         BOOLEAN
             );

             CREATE INDEX IF NOT EXISTS idx_run_query ON queries (run_id, query_name);
             CREATE INDEX IF NOT EXISTS idx_jit ON queries (jit_enabled);
             CREATE INDEX IF NOT EXISTS idx_status ON queries (status);

             CREATE TABLE IF NOT EXISTS profiling
             (
                 profile_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                 query_id      INTEGER NOT NULL,
                 cpu_data_lz4  BLOB,
                 heap_data_lz4 BLOB,
                 FOREIGN KEY (query_id) REFERENCES queries (query_id)
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
    try:
        with open(sql_file, 'r') as f:
            sql = f.read()

        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customer")
            count = cur.fetchone()[0]
            print(f"Loaded {count} customers")

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


def run_query_with_metrics(conn, query_file, jit_enabled, iteration, profile_enabled=False, heap_enabled=False,
                           fg_dir=None, output_dir=None, db_conn=None):
    query_name = query_file.stem

    with open(query_file, 'r') as f:
        query_sql = f.read()

    process = psutil.Process()
    io_start = process.io_counters()
    cpu_start = process.cpu_times()

    result = {
        'query_name': query_name,
        'iteration': iteration,
        'jit_enabled': jit_enabled,
        'status': 'SUCCESS',
        'error_message': None,
        'duration_ms': 0,
        'row_count': 0
    }

    detailed_metrics = {}
    timeseries = []
    sampling_active = True
    perf_proc = None
    perf_file = None
    massif_proc = None
    massif_file = None

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
            cur.execute("LOAD 'pgx_lower.so'")
            cur.execute(f"SET pgx_lower.enabled = {'true' if jit_enabled else 'false'}")

            # Get backend PID for profiling
            backend_pid = None
            if profile_enabled or heap_enabled:
                cur.execute("SELECT pg_backend_pid()")
                backend_pid = cur.fetchone()[0]

            if profile_enabled and backend_pid:
                jit_str = 'on' if jit_enabled else 'off'
                perf_file = str(output_dir / 'perf' / f'{query_name}_jit_{jit_str}.data')
                perf_proc = subprocess.Popen([
                    'sudo', 'perf', 'record',
                    '-p', str(backend_pid),
                    '-F', '99',
                    '-g',
                    '-o', perf_file
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                time.sleep(0.2)

            if heap_enabled and backend_pid:
                jit_str = 'on' if jit_enabled else 'off'
                massif_file = str(output_dir / 'heap' / f'{query_name}_jit_{jit_str}.data')
                massif_proc = subprocess.Popen([
                    'sudo', 'perf', 'record',
                    '-e', 'syscalls:sys_enter_mmap,syscalls:sys_enter_brk,page-faults',
                    '-p', str(backend_pid),
                    '-g',
                    '-o', massif_file
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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

            # Stop profilers after query execution completes
            if perf_proc is not None:
                try:
                    perf_proc.send_signal(signal.SIGINT)
                    perf_proc.wait(timeout=5)

                    username = os.environ.get('USER') or os.environ.get('LOGNAME')
                    if username and os.path.exists(perf_file):
                        subprocess.run(['sudo', 'chown', username, perf_file],
                                       stderr=subprocess.DEVNULL, check=False)
                        subprocess.run(['sudo', 'chmod', '644', perf_file],
                                       stderr=subprocess.DEVNULL, check=False)
                except:
                    try:
                        perf_proc.kill()
                    except:
                        pass

            if massif_proc is not None:
                try:
                    massif_proc.send_signal(signal.SIGINT)
                    massif_proc.wait(timeout=5)

                    username = os.environ.get('USER') or os.environ.get('LOGNAME')
                    if username and os.path.exists(massif_file):
                        subprocess.run(['sudo', 'chown', username, massif_file],
                                       stderr=subprocess.DEVNULL, check=False)
                        subprocess.run(['sudo', 'chmod', '644', massif_file],
                                       stderr=subprocess.DEVNULL, check=False)
                except:
                    try:
                        massif_proc.kill()
                    except:
                        pass

        conn.commit()

    except Exception as e:
        result['status'] = 'ERROR'
        result['error_message'] = str(e)
        conn.rollback()

        # Ensure profilers are stopped on error
        if perf_proc is not None:
            try:
                perf_proc.kill()
            except:
                pass

        if massif_proc is not None:
            try:
                massif_proc.kill()
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
    result['perf_file'] = perf_file
    result['massif_file'] = massif_file

    return result


def insert_metrics(db_conn, run_id, result):
    cursor = db_conn.cursor()
    cursor.execute("""
                   INSERT INTO queries (run_id, query_name, iteration, jit_enabled,
                                        status, duration_ms, row_count, error_message, metrics_json, postgres_metrics,
                                        timeseries_metrics, result_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   """, (
                       run_id,
                       result['query_name'],
                       result['iteration'],
                       result['jit_enabled'],
                       result['status'],
                       result['duration_ms'],
                       result['row_count'],
                       result['error_message'],
                       result['metrics_json'],
                       result.get('postgres_metrics'),
                       result['timeseries_metrics'],
                       result.get('result_hash')
                   ))
    db_conn.commit()
    return cursor.lastrowid


def ensure_flamegraph_tools(script_dir):
    fg_dir = script_dir / 'tools' / 'FlameGraph'
    if not fg_dir.exists():
        print("Cloning FlameGraph tools...")
        fg_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            'git', 'clone',
            'https://github.com/brendangregg/FlameGraph',
            str(fg_dir)
        ], check=True, capture_output=True)
        print("FlameGraph tools ready")
    return fg_dir


def process_profiles(perf_file, query_name, jit, fg_dir, output_dir):
    try:
        jit_str = 'on' if jit else 'off'
        svg_name = f"{query_name}_jit_{jit_str}.svg"
        svg_path = output_dir / 'profiles_cpu_latest' / svg_name

        # perf script | stackcollapse-perf.pl | flamegraph.pl > svg
        with open(svg_path, 'w') as svg_file:
            perf_script = subprocess.Popen(
                ['sudo', 'perf', 'script', '-f', '-i', perf_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            stackcollapse = subprocess.Popen(
                [str(fg_dir / 'stackcollapse-perf.pl')],
                stdin=perf_script.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            perf_script.stdout.close()

            title = f"{query_name} CPU JIT={'ON' if jit else 'OFF'}"
            subprocess.run(
                [str(fg_dir / 'flamegraph.pl'), '--title', title],
                stdin=stackcollapse.stdout,
                stdout=svg_file,
                stderr=subprocess.DEVNULL
            )
            stackcollapse.stdout.close()

        # Read and compress perf.data file (sudo cat since file is owned by root)
        result = subprocess.run(['sudo', 'cat', perf_file], capture_output=True, check=True)
        perf_data = result.stdout
        compressed_data = lz4.frame.compress(perf_data, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)

        return compressed_data

    except Exception as e:
        # Don't fail the benchmark if profile processing fails
        print(f"[cpu profile error: {e}]", end=' ')
        return None


def process_heap_profiles(mem_file, query_name, jit, fg_dir, output_dir):
    try:
        jit_str = 'on' if jit else 'off'
        svg_name = f"{query_name}_jit_{jit_str}.svg"
        svg_path = output_dir / 'profiles_heap_latest' / svg_name

        # perf mem report | convert to flamegraph format
        with open(svg_path, 'w') as svg_file:
            perf_script = subprocess.Popen(
                ['sudo', 'perf', 'script', '-f', '-i', mem_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            stackcollapse = subprocess.Popen(
                [str(fg_dir / 'stackcollapse-perf.pl')],
                stdin=perf_script.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            perf_script.stdout.close()

            title = f"{query_name} Memory JIT={'ON' if jit else 'OFF'}"
            subprocess.run(
                [str(fg_dir / 'flamegraph.pl'), '--title', title, '--countname', 'samples'],
                stdin=stackcollapse.stdout,
                stdout=svg_file,
                stderr=subprocess.DEVNULL
            )
            stackcollapse.stdout.close()

        # Read and compress perf.data file
        result = subprocess.run(['sudo', 'cat', mem_file], capture_output=True, check=True)
        mem_data = result.stdout
        compressed_data = lz4.frame.compress(mem_data, compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)

        return compressed_data

    except Exception as e:
        print(f"[heap profile error: {e}]", end=' ')
        return None


def timeout_handler():
    print("Starting timeout")
    time.sleep(300)
    os.kill(os.getpid(), signal.SIGKILL)


def setup_profiling_dirs(output_dir, script_dir, heap_enabled=False):
    fg_dir = ensure_flamegraph_tools(script_dir)

    dirs = ['profiles_cpu_latest', 'perf']
    if heap_enabled:
        dirs.extend(['profiles_heap_latest', 'heap'])

    for dirname in dirs:
        dirpath = output_dir / dirname
        if dirpath.exists():
            shutil.rmtree(dirpath)
        dirpath.mkdir()

    return fg_dir


def init_databases(sf, run_id, output_dir):
    db_file = output_dir / 'benchmark.db'
    db_conn = sqlite3.connect(db_file)
    db_conn.executescript(SCHEMA_SQL)

    pg_conn = psycopg2.connect(host='localhost', database='postgres', user='postgres')
    pg_conn.autocommit = False

    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT INTO runs (run_id, timestamp, scale_factor, iterations, postgres_version, pgx_version, hostname) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, int(time.time()), sf, 1, get_postgres_version(pg_conn), get_pgx_version(),
         subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()))
    db_conn.commit()

    return db_conn, pg_conn, db_file


def validate_results(db_conn, run_id):
    print(f"\nValidating results...")
    cursor = db_conn.cursor()

    cursor.execute("""
        SELECT query_name,
               MAX(CASE WHEN jit_enabled = 0 THEN result_hash END) AS postgres_hash,
               MAX(CASE WHEN jit_enabled = 1 THEN result_hash END) AS pgx_hash
        FROM queries
        WHERE run_id = ?
          AND status = 'SUCCESS'
        GROUP BY query_name
    """, (run_id,))

    mismatches = []
    for row in cursor.fetchall():
        query_name, postgres_hash, pgx_hash = row
        is_valid = (postgres_hash == pgx_hash) if (postgres_hash and pgx_hash) else None

        cursor.execute("""
            UPDATE queries
            SET hash_valid = ?
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


def run_benchmark_queries(pg_conn, db_conn, run_id, script_dir, profile_enabled, heap_enabled, fg_dir, output_dir):
    query_files = sorted((script_dir / 'queries').glob('q*.sql'))
    total = len(query_files) * 2
    current = 0

    print(f"\n{total} queries...")

    for jit in [False, True]:
        jit_mode = 'ON ' if jit else 'OFF'
        print(f"\n{'pgx-lower' if jit else 'PostgreSQL'} (JIT {jit_mode}):")

        for qf in query_files:
            current += 1
            print(f"[{current}/{total}] {qf.stem}...", end=' ', flush=True)

            metrics = run_query_with_metrics(
                pg_conn, qf, jit, 1,
                profile_enabled=profile_enabled,
                heap_enabled=heap_enabled,
                fg_dir=fg_dir,
                output_dir=output_dir,
                db_conn=db_conn
            )
            query_id = insert_metrics(db_conn, run_id, metrics)

            cpu_blob = None
            heap_blob = None

            if profile_enabled and metrics['status'] == 'SUCCESS' and metrics.get('perf_file'):
                cpu_blob = process_profiles(
                    metrics['perf_file'],
                    metrics['query_name'],
                    jit,
                    fg_dir,
                    output_dir
                )

            if heap_enabled and metrics['status'] == 'SUCCESS' and metrics.get('massif_file'):
                heap_blob = process_heap_profiles(
                    metrics['massif_file'],
                    metrics['query_name'],
                    jit,
                    fg_dir,
                    output_dir
                )

            if cpu_blob is not None or heap_blob is not None:
                cursor = db_conn.cursor()
                cursor.execute("""
                    INSERT INTO profiling (query_id, cpu_data_lz4, heap_data_lz4)
                    VALUES (?, ?, ?)
                """, (query_id, cpu_blob, heap_blob))
                db_conn.commit()

            if metrics['status'] == 'SUCCESS':
                hash_short = metrics['result_hash'][:8]
                print(f"{metrics['duration_ms']:.1f}ms [{hash_short}]")
            else:
                print(f"ERROR: {metrics['error_message']}")


def main():
    threading.Thread(target=timeout_handler, daemon=True).start()

    parser = argparse.ArgumentParser(description='Run TPC-H benchmarks for pgx-lower')
    parser.add_argument('scale_factor', type=float, nargs='?', default=0.01,
                        help='TPC-H scale factor (default: 0.01)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable CPU profiling with perf and flame graphs')
    parser.add_argument('--heap', action='store_true',
                        help='Enable heap profiling with Valgrind massif')
    args = parser.parse_args()

    sf = args.scale_factor
    profile_enabled = args.profile
    heap_enabled = args.heap

    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    fg_dir = None
    if profile_enabled or heap_enabled:
        fg_dir = setup_profiling_dirs(output_dir, script_dir, heap_enabled)

    run_id = f"pgx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_sf{int(sf * 100):03d}"
    print(f"{run_id} (SF={sf})")

    db_conn, pg_conn, db_file = init_databases(sf, run_id, output_dir)

    if not generate_tpch_data(sf):
        sys.exit(1)

    if not load_tpch_database(pg_conn):
        sys.exit(1)

    run_benchmark_queries(pg_conn, db_conn, run_id, script_dir, profile_enabled, heap_enabled, fg_dir, output_dir)

    pg_conn.close()
    validate_results(db_conn, run_id)
    db_conn.close()

    print(f"\nDone: {db_file}")


if __name__ == '__main__':
    main()
