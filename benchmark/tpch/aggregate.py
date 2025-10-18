import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Optional


def _insert_aggregate_stats(cursor: sqlite3.Cursor, bench_stats: dict, profile_stats: dict) -> None:
    cursor.execute("""
                   INSERT INTO aggregate_benchmarks (pgx_version, postgres_version, scale_factor, pgx_enabled,
                                                     run_timestamp,
                                                     execution_metadata, metrics_json, postgres_metrics)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   """, (
                       bench_stats['pgx_version'],
                       bench_stats['postgres_version'],
                       bench_stats['scale_factor'],
                       bench_stats['pgx_enabled'],
                       bench_stats['run_timestamp'],
                       bench_stats['execution_metadata'],
                       bench_stats['metrics_json'],
                       bench_stats['postgres_metrics']
                   ))

    cursor.execute("""
                   INSERT INTO aggregate_profiles (pgx_version, postgres_version, scale_factor, pgx_enabled,
                                                   run_timestamp,
                                                   cpu_data_lz4, heap_data_lz4, cpu_flamegraph_lz4, heap_flamegraph_lz4,
                                                   profile_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   """, (
                       profile_stats['pgx_version'],
                       profile_stats['postgres_version'],
                       profile_stats['scale_factor'],
                       profile_stats['pgx_enabled'],
                       profile_stats['run_timestamp'],
                       profile_stats['cpu_data_lz4'],
                       profile_stats['heap_data_lz4'],
                       profile_stats['cpu_flamegraph_lz4'],
                       profile_stats['heap_flamegraph_lz4'],
                       profile_stats['profile_metadata']
                   ))


def percentile(data: list[float], p: float) -> Optional[float]:
    if not data:
        return None
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


def aggregate_benchmark_metrics(
        conn: sqlite3.Connection,
        pgx_version: str,
        postgres_version: str,
        scale_factor: float,
        pgx_enabled: bool,
        run_timestamp: str
) -> dict:
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT q.query_name, q.execution_metadata, q.metrics_json, q.postgres_metrics
                   FROM queries q
                            JOIN runs r ON q.run_id = r.run_id
                   WHERE r.run_timestamp = ?
                     AND q.pgx_enabled = ?
                   """, (run_timestamp, pgx_enabled))

    query_results = cursor.fetchall()

    query_names = []
    duration_times = []
    execution_times = []
    planning_times = []
    cache_hit_ratios = []
    disk_read_blocks = 0
    disk_write_blocks = 0
    spilled_mb = 0.0
    native_jit_count = 0
    native_jit_times = []

    io_read_mb_list = []
    io_write_mb_list = []
    cpu_user_sec_list = []
    cpu_system_sec_list = []
    memory_peak_mb_list = []

    successful_queries = 0
    failed_queries = 0

    for query_name, exec_meta_str, metrics_json_str, pg_metrics_str in query_results:
        query_names.append(query_name)

        if exec_meta_str:
            exec_meta = json.loads(exec_meta_str)
            if exec_meta.get('status') == 'SUCCESS':
                successful_queries += 1
                duration_times.append(exec_meta.get('duration_ms', 0))
            else:
                failed_queries += 1

        if metrics_json_str:
            py_metrics = json.loads(metrics_json_str)
            io_read_mb_list.append(py_metrics.get('io_read_mb', 0))
            io_write_mb_list.append(py_metrics.get('io_write_mb', 0))
            cpu_user_sec_list.append(py_metrics.get('cpu_user_sec', 0))
            cpu_system_sec_list.append(py_metrics.get('cpu_system_sec', 0))
            memory_peak_mb_list.append(py_metrics.get('memory_peak_mb', 0))

        if pg_metrics_str:
            pg_metrics = json.loads(pg_metrics_str)

            execution_times.append(pg_metrics.get('execution_time_ms', 0))
            planning_times.append(pg_metrics.get('planning_time_ms', 0))

            if 'cache_hit_ratio' in pg_metrics:
                cache_hit_ratios.append(pg_metrics['cache_hit_ratio'])

            buffers = pg_metrics.get('buffers', {})
            disk_read_blocks += buffers.get('shared_read', 0) + buffers.get('local_read', 0)
            disk_write_blocks += buffers.get('shared_written', 0) + buffers.get('local_written', 0)

            if pg_metrics.get('spilled_to_disk'):
                spilled_mb += (buffers.get('temp_written', 0) * 8) / 1024

            if 'jit' in pg_metrics:
                native_jit_count += 1
                jit_time = pg_metrics['jit'].get('Generation Time', 0)
                if jit_time > 0:
                    native_jit_times.append(jit_time)

    execution_metadata_json = json.dumps({
        'total_queries': len(query_results),
        'successful_queries': successful_queries,
        'failed_queries': failed_queries,
        'query_list': query_names
    })

    metrics_json = json.dumps({
        'duration_ms': {
            'avg': statistics.mean(duration_times) if duration_times else None,
            'median': percentile(duration_times, 0.5),
            'p95': percentile(duration_times, 0.95),
            'p99': percentile(duration_times, 0.99),
            'min': min(duration_times) if duration_times else None,
            'max': max(duration_times) if duration_times else None,
            'total': sum(duration_times) if duration_times else 0
        },
        'io_read_mb': {
            'avg': statistics.mean(io_read_mb_list) if io_read_mb_list else None,
            'total': sum(io_read_mb_list) if io_read_mb_list else 0
        },
        'io_write_mb': {
            'avg': statistics.mean(io_write_mb_list) if io_write_mb_list else None,
            'total': sum(io_write_mb_list) if io_write_mb_list else 0
        },
        'cpu_user_sec': {
            'avg': statistics.mean(cpu_user_sec_list) if cpu_user_sec_list else None,
            'total': sum(cpu_user_sec_list) if cpu_user_sec_list else 0
        },
        'cpu_system_sec': {
            'avg': statistics.mean(cpu_system_sec_list) if cpu_system_sec_list else None,
            'total': sum(cpu_system_sec_list) if cpu_system_sec_list else 0
        },
        'memory_peak_mb': {
            'avg': statistics.mean(memory_peak_mb_list) if memory_peak_mb_list else None,
            'max': max(memory_peak_mb_list) if memory_peak_mb_list else None
        }
    })

    postgres_metrics_json = json.dumps({
        'execution_time_ms': {
            'avg': statistics.mean(execution_times) if execution_times else None,
            'median': percentile(execution_times, 0.5),
            'p95': percentile(execution_times, 0.95),
            'p99': percentile(execution_times, 0.99),
            'min': min(execution_times) if execution_times else None,
            'max': max(execution_times) if execution_times else None,
            'total': sum(execution_times) if execution_times else 0
        },
        'planning_time_ms': {
            'avg': statistics.mean(planning_times) if planning_times else None,
            'total': sum(planning_times) if planning_times else 0
        },
        'cache_hit_ratio': {
            'avg': statistics.mean(cache_hit_ratios) if cache_hit_ratios else None
        },
        'disk_blocks': {
            'read': disk_read_blocks,
            'write': disk_write_blocks
        },
        'spilled_mb': spilled_mb,
        'native_jit': {
            'query_count': native_jit_count,
            'avg_time_ms': statistics.mean(native_jit_times) if native_jit_times else None
        }
    })

    return {
        'pgx_version': pgx_version,
        'postgres_version': postgres_version,
        'scale_factor': scale_factor,
        'pgx_enabled': pgx_enabled,
        'run_timestamp': run_timestamp,
        'execution_metadata': execution_metadata_json,
        'metrics_json': metrics_json,
        'postgres_metrics': postgres_metrics_json
    }


def aggregate_profile_metrics(
        conn: sqlite3.Connection,
        pgx_version: str,
        postgres_version: str,
        scale_factor: float,
        pgx_enabled: bool,
        run_timestamp: str
) -> dict:
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT p.cpu_data_lz4,
                          p.heap_data_lz4,
                          p.cpu_flamegraph_lz4,
                          p.heap_flamegraph_lz4,
                          p.profile_metadata
                   FROM profiling p
                            JOIN queries q ON p.query_id = q.query_id
                            JOIN runs r ON q.run_id = r.run_id
                   WHERE r.run_timestamp = ?
                     AND q.pgx_enabled = ?
                   """, (run_timestamp, pgx_enabled))

    profile_results = cursor.fetchall()

    cpu_count = 0
    heap_count = 0
    total_cpu_raw_kb = 0.0
    total_cpu_compressed_kb = 0.0
    total_heap_raw_kb = 0.0
    total_heap_compressed_kb = 0.0
    cpu_blobs = []
    heap_blobs = []
    cpu_flamegraph_blobs = []
    heap_flamegraph_blobs = []

    for cpu_blob, heap_blob, cpu_fg, heap_fg, profile_metadata_str in profile_results:
        if profile_metadata_str:
            meta = json.loads(profile_metadata_str)

            if cpu_blob:
                cpu_count += 1
                cpu_blobs.append(cpu_blob)
                if cpu_fg:
                    cpu_flamegraph_blobs.append(cpu_fg)
                cpu_meta = meta.get('cpu', {})
                total_cpu_raw_kb += cpu_meta.get('raw_kb', 0)
                total_cpu_compressed_kb += cpu_meta.get('compressed_kb', 0)

            if heap_blob:
                heap_count += 1
                heap_blobs.append(heap_blob)
                if heap_fg:
                    heap_flamegraph_blobs.append(heap_fg)
                heap_meta = meta.get('heap', {})
                total_heap_raw_kb += heap_meta.get('raw_kb', 0)
                total_heap_compressed_kb += heap_meta.get('compressed_kb', 0)

    # Merge blobs (simple - take first one. TODO: implement proper merging)
    merged_cpu_blob = cpu_blobs[0] if cpu_blobs else None
    merged_heap_blob = heap_blobs[0] if heap_blobs else None
    merged_cpu_flamegraph = cpu_flamegraph_blobs[0] if cpu_flamegraph_blobs else None
    merged_heap_flamegraph = heap_flamegraph_blobs[0] if heap_flamegraph_blobs else None

    # Build profile_metadata JSON
    cpu_compression_ratio = (
        total_cpu_raw_kb / total_cpu_compressed_kb
        if total_cpu_compressed_kb > 0 else None
    )
    heap_compression_ratio = (
        total_heap_raw_kb / total_heap_compressed_kb
        if total_heap_compressed_kb > 0 else None
    )

    profile_metadata_json = json.dumps({
        'cpu': {
            'total_profiles': cpu_count,
            'raw_kb': total_cpu_raw_kb,
            'compressed_kb': total_cpu_compressed_kb,
            'compression_ratio': cpu_compression_ratio,
            'per_query_storage_kb': total_cpu_compressed_kb / cpu_count if cpu_count > 0 else None
        },
        'heap': {
            'total_profiles': heap_count,
            'raw_kb': total_heap_raw_kb,
            'compressed_kb': total_heap_compressed_kb,
            'compression_ratio': heap_compression_ratio,
            'per_query_storage_kb': total_heap_compressed_kb / heap_count if heap_count > 0 else None
        }
    })

    return {
        'pgx_version': pgx_version,
        'postgres_version': postgres_version,
        'scale_factor': scale_factor,
        'pgx_enabled': pgx_enabled,
        'run_timestamp': run_timestamp,
        'cpu_data_lz4': merged_cpu_blob,
        'heap_data_lz4': merged_heap_blob,
        'cpu_flamegraph_lz4': merged_cpu_flamegraph,
        'heap_flamegraph_lz4': merged_heap_flamegraph,
        'profile_metadata': profile_metadata_json
    }


def compute_all_aggregates(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM aggregate_benchmarks")
    cursor.execute("DELETE FROM aggregate_profiles")

    cursor.execute("""
                   SELECT r.pgx_version,
                          r.postgres_version,
                          r.scale_factor,
                          q.pgx_enabled,
                          MAX(r.run_timestamp) as latest_timestamp
                   FROM queries q
                            JOIN runs r ON q.run_id = r.run_id
                   GROUP BY r.pgx_version, r.postgres_version, r.scale_factor, q.pgx_enabled
                   """)

    configurations = cursor.fetchall()

    print(f"Found {len(configurations)} unique configurations")

    for pgx_ver, pg_ver, scale, pgx_on, timestamp in configurations:
        print(f"Aggregating: {pgx_ver[:8]} / PG {pg_ver} / SF {scale} / "
              f"pgx={'on' if pgx_on else 'off'} / latest={timestamp}")

        bench_stats = aggregate_benchmark_metrics(conn, pgx_ver, pg_ver, scale, pgx_on, timestamp)
        profile_stats = aggregate_profile_metrics(conn, pgx_ver, pg_ver, scale, pgx_on, timestamp)
        _insert_aggregate_stats(cursor, bench_stats, profile_stats)

    conn.commit()
    conn.close()

    print(f"Aggregation complete: {len(configurations)} configurations")


def aggregate_latest_run(db_path: Path, run_timestamp: str) -> None:
    compute_all_aggregates(db_path)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python aggregate.py <db_path>              # Recompute all aggregates")
        print("  python aggregate.py <db_path> <timestamp>  # Aggregate specific run")
        sys.exit(1)

    db_path = Path(sys.argv[1])

    if len(sys.argv) == 3:
        aggregate_latest_run(db_path, sys.argv[2])
    else:
        compute_all_aggregates(db_path)
