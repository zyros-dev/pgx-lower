import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Optional


def _insert_aggregate_stats(cursor: sqlite3.Cursor, bench_stats: dict, profile_stats: dict) -> None:
    cursor.execute("""
        INSERT INTO aggregate_benchmarks (
            pgx_version, postgres_version, scale_factor, pgx_enabled, run_timestamp,
            total_queries, successful_queries, failed_queries,
            total_execution_time_ms, avg_execution_time_ms, median_execution_time_ms,
            p95_execution_time_ms, p99_execution_time_ms,
            min_execution_time_ms, max_execution_time_ms,
            avg_planning_time_ms, total_planning_time_ms,
            avg_cache_hit_ratio, total_disk_read_blocks, total_disk_write_blocks,
            total_spilled_mb, queries_with_native_jit, avg_native_jit_time_ms,
            avg_peak_memory_mb, max_peak_memory_mb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tuple(bench_stats.values()))

    cursor.execute("""
        INSERT INTO aggregate_profiles (
            pgx_version, postgres_version, scale_factor, pgx_enabled, run_timestamp,
            cpu_profiles_count, heap_profiles_count,
            total_cpu_profile_raw_kb, total_cpu_profile_compressed_kb, avg_cpu_compression_ratio,
            total_heap_profile_raw_kb, total_heap_profile_compressed_kb, avg_heap_compression_ratio,
            total_storage_kb, storage_per_query_kb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tuple(profile_stats.values()))


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
    """
    Compute aggregate benchmark statistics for a specific configuration.

    Returns dict of aggregated metrics.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT query_name, postgres_metrics
        FROM queries
        WHERE run_timestamp = ? AND pgx_enabled = ?
    """, (run_timestamp, pgx_enabled))

    query_results = cursor.fetchall()

    # Parse metrics from JSON
    execution_times = []
    planning_times = []
    cache_hit_ratios = []
    disk_read_blocks = 0
    disk_write_blocks = 0
    spilled_mb = 0.0
    native_jit_count = 0
    native_jit_times = []
    peak_memories_mb = []

    for query_name, metrics_json in query_results:
        if not metrics_json:
            continue

        m = json.loads(metrics_json)

        # Timing
        execution_times.append(m.get('execution_time_ms', 0))
        planning_times.append(m.get('planning_time_ms', 0))

        # I/O
        if 'cache_hit_ratio' in m:
            cache_hit_ratios.append(m['cache_hit_ratio'])

        buffers = m.get('buffers', {})
        disk_read_blocks += buffers.get('shared_read', 0) + buffers.get('local_read', 0)
        disk_write_blocks += buffers.get('shared_written', 0) + buffers.get('local_written', 0)

        if m.get('spilled_to_disk'):
            spilled_mb += (buffers.get('temp_written', 0) * 8) / 1024  # blocks to MB

        # PostgreSQL native JIT (orthogonal to pgx_enabled)
        if 'jit' in m:
            native_jit_count += 1
            jit_time = m['jit'].get('Generation Time', 0)
            if jit_time > 0:
                native_jit_times.append(jit_time)

        # Memory
        if 'memory_contexts' in m:
            contexts = m['memory_contexts']
            total_used = sum(
                ctx.get('used_bytes', 0)
                for ctx in contexts.values()
                if isinstance(ctx, dict)
            )
            peak_memories_mb.append(total_used / (1024 * 1024))

    # Compute statistics
    stats = {
        'pgx_version': pgx_version,
        'postgres_version': postgres_version,
        'scale_factor': scale_factor,
        'pgx_enabled': pgx_enabled,
        'run_timestamp': run_timestamp,

        'total_queries': len(query_results),
        'successful_queries': len(execution_times),  # Has metrics = succeeded
        'failed_queries': len(query_results) - len(execution_times),

        'total_execution_time_ms': sum(execution_times) if execution_times else 0,
        'avg_execution_time_ms': statistics.mean(execution_times) if execution_times else None,
        'median_execution_time_ms': percentile(execution_times, 0.5),
        'p95_execution_time_ms': percentile(execution_times, 0.95),
        'p99_execution_time_ms': percentile(execution_times, 0.99),
        'min_execution_time_ms': min(execution_times) if execution_times else None,
        'max_execution_time_ms': max(execution_times) if execution_times else None,

        'avg_planning_time_ms': statistics.mean(planning_times) if planning_times else None,
        'total_planning_time_ms': sum(planning_times) if planning_times else 0,

        'avg_cache_hit_ratio': statistics.mean(cache_hit_ratios) if cache_hit_ratios else None,
        'total_disk_read_blocks': disk_read_blocks,
        'total_disk_write_blocks': disk_write_blocks,
        'total_spilled_mb': spilled_mb,

        'queries_with_native_jit': native_jit_count,
        'avg_native_jit_time_ms': statistics.mean(native_jit_times) if native_jit_times else None,

        'avg_peak_memory_mb': statistics.mean(peak_memories_mb) if peak_memories_mb else None,
        'max_peak_memory_mb': max(peak_memories_mb) if peak_memories_mb else None,
    }

    return stats


def aggregate_profile_metrics(
        conn: sqlite3.Connection,
        pgx_version: str,
        postgres_version: str,
        scale_factor: float,
        pgx_enabled: bool,
        run_timestamp: str
) -> dict:
    """
    Compute aggregate profiling statistics for a specific configuration.

    Returns dict of aggregated metrics.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            cpu_data_lz4,
            heap_data_lz4,
            cpu_raw_size_kb,
            cpu_compressed_size_kb,
            heap_raw_size_kb,
            heap_compressed_size_kb
        FROM profiling
        WHERE run_timestamp = ? AND pgx_enabled = ?
    """, (run_timestamp, pgx_enabled))

    profile_results = cursor.fetchall()

    cpu_count = 0
    heap_count = 0
    total_cpu_raw_kb = 0.0
    total_cpu_compressed_kb = 0.0
    total_heap_raw_kb = 0.0
    total_heap_compressed_kb = 0.0

    for cpu_blob, heap_blob, cpu_raw, cpu_comp, heap_raw, heap_comp in profile_results:
        if cpu_blob:
            cpu_count += 1
            total_cpu_raw_kb += cpu_raw or 0
            total_cpu_compressed_kb += cpu_comp or 0

        if heap_blob:
            heap_count += 1
            total_heap_raw_kb += heap_raw or 0
            total_heap_compressed_kb += heap_comp or 0

    # Compute compression ratios
    cpu_compression_ratio = (
        total_cpu_raw_kb / total_cpu_compressed_kb
        if total_cpu_compressed_kb > 0 else None
    )

    heap_compression_ratio = (
        total_heap_raw_kb / total_heap_compressed_kb
        if total_heap_compressed_kb > 0 else None
    )

    total_storage_kb = total_cpu_compressed_kb + total_heap_compressed_kb
    storage_per_query_kb = total_storage_kb / len(profile_results) if profile_results else None

    stats = {
        'pgx_version': pgx_version,
        'postgres_version': postgres_version,
        'scale_factor': scale_factor,
        'pgx_enabled': pgx_enabled,
        'run_timestamp': run_timestamp,

        'cpu_profiles_count': cpu_count,
        'heap_profiles_count': heap_count,

        'total_cpu_profile_raw_kb': total_cpu_raw_kb,
        'total_cpu_profile_compressed_kb': total_cpu_compressed_kb,
        'avg_cpu_compression_ratio': cpu_compression_ratio,

        'total_heap_profile_raw_kb': total_heap_raw_kb,
        'total_heap_profile_compressed_kb': total_heap_compressed_kb,
        'avg_heap_compression_ratio': heap_compression_ratio,

        'total_storage_kb': total_storage_kb,
        'storage_per_query_kb': storage_per_query_kb,
    }

    return stats


def compute_all_aggregates(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM aggregate_benchmarks")
    cursor.execute("DELETE FROM aggregate_profiles")

    cursor.execute("""
        SELECT
            pgx_version,
            postgres_version,
            scale_factor,
            pgx_enabled,
            MAX(run_timestamp) as latest_timestamp
        FROM queries
        GROUP BY pgx_version, postgres_version, scale_factor, pgx_enabled
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
