#!/usr/bin/env python3
"""
Benchmark Visualization Dashboard Server

Lightweight FastAPI server for browsing pgx-lower TPC-H benchmark results.
Serves static frontend and provides REST API to query SQLite database.
"""

import sqlite3
import lz4.frame
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

# Paths
WEBSITE_DIR = Path(__file__).parent
DB_PATH = WEBSITE_DIR.parent / "output" / "benchmark.db"
STATIC_DIR = WEBSITE_DIR / "static"

# FastAPI app
app = FastAPI(
    title="pgx-lower Benchmark Dashboard",
    description="Visualize TPC-H benchmark results with flame charts and performance analysis",
    version="1.0.0"
)

# CORS for LAN access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    """Get database connection with read-only access."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row  # Return dicts instead of tuples
    return conn


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/overview")
async def get_overview(
    scale_factor: Optional[float] = None,
    run_timestamp: Optional[str] = None
):
    """
    Get high-level overview metrics.

    Returns:
        - pgx_avg: Average execution time with pgx enabled (ms)
        - native_avg: Average execution time native PostgreSQL (ms)
        - speedup: Ratio native/pgx
        - total_runs: Number of distinct runs
        - scale_factor: Selected scale factor
    """
    conn = get_db()
    cursor = conn.cursor()

    # If no run specified, get latest
    if not run_timestamp:
        cursor.execute("SELECT run_timestamp FROM queries ORDER BY run_timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            run_timestamp = row[0]

    # If no scale specified, get from latest run
    if not scale_factor:
        cursor.execute("SELECT scale_factor FROM queries WHERE run_timestamp = ? LIMIT 1", (run_timestamp,))
        row = cursor.fetchone()
        if row:
            scale_factor = row[0]

    # Get aggregates for both pgx and native
    cursor.execute("""
        SELECT pgx_enabled, avg_execution_time_ms, successful_queries
        FROM aggregate_benchmarks
        WHERE scale_factor = ? AND run_timestamp = ?
    """, (scale_factor, run_timestamp))

    results = cursor.fetchall()

    pgx_avg = None
    native_avg = None

    for row in results:
        if row[0]:  # pgx_enabled
            pgx_avg = row[1]
        else:
            native_avg = row[1]

    speedup = None
    if pgx_avg and native_avg and pgx_avg > 0:
        speedup = round(native_avg / pgx_avg, 2)

    # Count total distinct runs
    cursor.execute("SELECT COUNT(DISTINCT run_timestamp) FROM queries")
    total_runs = cursor.fetchone()[0]

    conn.close()

    return {
        "pgx_avg": round(pgx_avg, 2) if pgx_avg else None,
        "native_avg": round(native_avg, 2) if native_avg else None,
        "speedup": speedup,
        "total_runs": total_runs,
        "scale_factor": scale_factor,
        "run_timestamp": run_timestamp
    }


@app.get("/api/queries")
async def get_queries(
    pgx_enabled: Optional[bool] = None,
    scale_factor: Optional[float] = None,
    run_timestamp: Optional[str] = None
):
    """
    Get individual query execution data.

    Query params:
        - pgx_enabled: Filter by pgx enabled/disabled
        - scale_factor: Filter by scale factor
        - run_timestamp: Filter by run timestamp (defaults to latest)

    Returns: List of query results with execution metrics
    """
    conn = get_db()
    cursor = conn.cursor()

    # Build dynamic query
    conditions = []
    params = []

    # Handle "latest" as None
    if run_timestamp and run_timestamp != "latest":
        conditions.append("run_timestamp = ?")
        params.append(run_timestamp)
    else:
        # Get latest run
        cursor.execute("SELECT run_timestamp FROM queries ORDER BY run_timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            conditions.append("run_timestamp = ?")
            params.append(row[0])

    if pgx_enabled is not None:
        conditions.append("pgx_enabled = ?")
        params.append(pgx_enabled)

    if scale_factor is not None:
        conditions.append("scale_factor = ?")
        params.append(scale_factor)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT
            query_name,
            pgx_enabled,
            status,
            duration_ms,
            row_count,
            postgres_metrics,
            run_timestamp,
            scale_factor,
            pgx_version,
            postgres_version
        FROM queries
        WHERE {where_clause}
        ORDER BY query_name, pgx_enabled
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Convert to dict and parse postgres_metrics JSON
    results = []
    for row in rows:
        import json

        metrics = json.loads(row[5]) if row[5] else {}

        results.append({
            "query_name": row[0],
            "pgx_enabled": bool(row[1]),
            "status": row[2],
            "duration_ms": round(row[3], 2) if row[3] else None,
            "row_count": row[4],
            "exec_time": round(metrics.get("execution_time_ms", 0), 2),
            "planning_time": round(metrics.get("planning_time_ms", 0), 2),
            "buffers": metrics.get("buffers", {}),
            "run_timestamp": row[6],
            "scale_factor": row[7],
            "pgx_version": row[8],
            "postgres_version": row[9]
        })

    conn.close()

    return results


@app.get("/api/aggregates")
async def get_aggregates(
    scale_factor: Optional[float] = None
):
    """
    Get aggregate benchmark statistics.

    Returns summary statistics grouped by configuration:
    (pgx_version, postgres_version, scale_factor, pgx_enabled)
    """
    conn = get_db()
    cursor = conn.cursor()

    if scale_factor:
        cursor.execute("""
            SELECT * FROM aggregate_benchmarks
            WHERE scale_factor = ?
            ORDER BY run_timestamp DESC, pgx_enabled DESC
        """, (scale_factor,))
    else:
        cursor.execute("""
            SELECT * FROM aggregate_benchmarks
            ORDER BY run_timestamp DESC, pgx_enabled DESC
        """)

    rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append(dict(row))

    conn.close()

    return results


@app.get("/api/runs")
async def get_runs():
    """
    Get list of all benchmark runs.

    Returns: List of runs with timestamp, scale factor, and query count
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            run_timestamp,
            scale_factor,
            COUNT(*) as query_count,
            SUM(CASE WHEN pgx_enabled THEN 1 ELSE 0 END) as pgx_count,
            SUM(CASE WHEN NOT pgx_enabled THEN 1 ELSE 0 END) as native_count
        FROM queries
        GROUP BY run_timestamp, scale_factor
        ORDER BY run_timestamp DESC
    """)

    rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append({
            "run_timestamp": row[0],
            "scale_factor": row[1],
            "query_count": row[2],
            "pgx_count": row[3],
            "native_count": row[4]
        })

    conn.close()

    return results


@app.get("/api/timeseries")
async def get_timeseries(
    query: str,
    metric: str = "exec_time"
):
    """
    Get time series data for a specific query and metric.

    Args:
        query: Query name (e.g., "Q1")
        metric: Metric to track (exec_time, planning_time, etc.)

    Returns: Time series with pgx and native values
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            run_timestamp,
            pgx_enabled,
            postgres_metrics
        FROM queries
        WHERE query_name = ?
        ORDER BY run_timestamp, pgx_enabled
    """, (query,))

    rows = cursor.fetchall()

    # Group by timestamp
    import json
    from collections import defaultdict

    data = defaultdict(lambda: {"run_timestamp": None, "pgx": None, "native": None})

    for row in rows:
        timestamp = row[0]
        pgx_enabled = row[1]
        metrics = json.loads(row[2]) if row[2] else {}

        value = None
        if metric == "exec_time":
            value = metrics.get("execution_time_ms", 0)
        elif metric == "planning_time":
            value = metrics.get("planning_time_ms", 0)

        data[timestamp]["run_timestamp"] = timestamp
        if pgx_enabled:
            data[timestamp]["pgx"] = round(value, 2) if value else None
        else:
            data[timestamp]["native"] = round(value, 2) if value else None

    conn.close()

    return sorted(data.values(), key=lambda x: x["run_timestamp"])


@app.get("/api/heatmap")
async def get_heatmap(
    scale_factor: Optional[float] = None,
    run_timestamp: Optional[str] = None
):
    """
    Get performance heatmap data (query × metric grid).

    Returns color-coded performance matrix with percentile-based thresholds.
    Each cell contains absolute value and color category (fast/medium/slow).

    Args:
        scale_factor: Filter by scale factor
        run_timestamp: Filter by run timestamp (defaults to latest)

    Returns:
        - queries: List of query names
        - metrics: List of metric names
        - data: Dict[query][metric] = {value, color, percentile}
        - thresholds: Dict[metric] = {p33, p67} for color boundaries
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get latest run if not specified
    if not run_timestamp or run_timestamp == "latest":
        cursor.execute("SELECT run_timestamp FROM queries ORDER BY run_timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            run_timestamp = row[0]

    # Get scale factor from latest run if not specified
    if not scale_factor:
        cursor.execute("SELECT scale_factor FROM queries WHERE run_timestamp = ? LIMIT 1", (run_timestamp,))
        row = cursor.fetchone()
        if row:
            scale_factor = row[0]

    # Fetch all queries for this run
    cursor.execute("""
        SELECT query_name, pgx_enabled, postgres_metrics
        FROM queries
        WHERE run_timestamp = ? AND scale_factor = ?
        ORDER BY query_name, pgx_enabled
    """, (run_timestamp, scale_factor))

    rows = cursor.fetchall()
    conn.close()

    # Parse metrics
    import json
    import numpy as np

    # Metrics to include in heatmap
    metric_keys = [
        ("exec_time", "execution_time_ms"),
        ("plan_time", "planning_time_ms"),
        ("memory", "peak_memory_kb"),
        ("cache_hit", "cache_hit_ratio"),
        ("disk_read", "shared_blks_read"),
        ("spilled", "temp_blks_written")
    ]

    # Collect all values per metric for percentile calculation
    metric_values = {key: [] for key, _ in metric_keys}
    query_data = {}

    for row in rows:
        query_name = row[0]
        pgx_enabled = row[1]
        metrics_json = json.loads(row[2]) if row[2] else {}

        # Store data per query+pgx combination
        key = f"{query_name}_{'pgx' if pgx_enabled else 'native'}"
        query_data[key] = {}

        for display_name, json_key in metric_keys:
            value = metrics_json.get(json_key)
            if value is not None:
                query_data[key][display_name] = value
                metric_values[display_name].append(value)

    # Calculate percentiles (33rd and 67th) for each metric
    thresholds = {}
    for metric_name, values in metric_values.items():
        if values:
            arr = np.array(values)
            p33 = np.percentile(arr, 33)
            p67 = np.percentile(arr, 67)
            thresholds[metric_name] = {"p33": float(p33), "p67": float(p67)}
        else:
            thresholds[metric_name] = {"p33": 0, "p67": 0}

    # Assign colors based on percentiles
    # Lower is better for exec_time, plan_time, memory, disk_read, spilled
    # Higher is better for cache_hit
    better_higher = {"cache_hit"}

    heatmap_data = {}
    for query_key, metrics in query_data.items():
        heatmap_data[query_key] = {}
        for metric_name, value in metrics.items():
            thresh = thresholds[metric_name]

            if metric_name in better_higher:
                # Higher is better
                if value >= thresh["p67"]:
                    color = "fast"
                    percentile = 67
                elif value >= thresh["p33"]:
                    color = "medium"
                    percentile = 50
                else:
                    color = "slow"
                    percentile = 33
            else:
                # Lower is better
                if value <= thresh["p33"]:
                    color = "fast"
                    percentile = 67
                elif value <= thresh["p67"]:
                    color = "medium"
                    percentile = 50
                else:
                    color = "slow"
                    percentile = 33

            heatmap_data[query_key][metric_name] = {
                "value": round(value, 2),
                "color": color,
                "percentile": percentile
            }

    # Get unique query names and metric names
    queries = sorted(set(k.rsplit('_', 1)[0] for k in query_data.keys()))
    metrics = [name for name, _ in metric_keys]

    return {
        "queries": queries,
        "metrics": metrics,
        "data": heatmap_data,
        "thresholds": thresholds,
        "scale_factor": scale_factor,
        "run_timestamp": run_timestamp
    }


@app.get("/api/profile/list")
async def get_profile_list(
    scale_factor: Optional[float] = None,
    run_timestamp: Optional[str] = None
):
    """Get list of available profiles with metadata."""
    conn = get_db()
    cursor = conn.cursor()

    # Build query with optional filters
    where_clauses = []
    params = []

    if scale_factor is not None:
        where_clauses.append("q.scale_factor = ?")
        params.append(scale_factor)

    if run_timestamp and run_timestamp != "latest":
        where_clauses.append("p.run_timestamp = ?")
        params.append(run_timestamp)

    where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

    cursor.execute(f"""
        SELECT
            q.query_name,
            p.pgx_enabled,
            p.run_timestamp,
            CASE WHEN p.cpu_data_lz4 IS NOT NULL THEN 1 ELSE 0 END as has_cpu,
            CASE WHEN p.heap_data_lz4 IS NOT NULL THEN 1 ELSE 0 END as has_heap,
            p.cpu_raw_size_kb,
            p.cpu_compressed_size_kb,
            p.heap_raw_size_kb,
            p.heap_compressed_size_kb
        FROM profiling p
        JOIN queries q ON p.query_id = q.query_id
        WHERE 1=1 {where_sql}
        ORDER BY q.query_name, p.pgx_enabled
    """, params)

    profiles = []
    for row in cursor.fetchall():
        profiles.append({
            "query": row[0],
            "pgx_enabled": bool(row[1]),
            "run_timestamp": row[2],
            "has_cpu": bool(row[3]),
            "has_heap": bool(row[4]),
            "cpu_raw_kb": row[5],
            "cpu_compressed_kb": row[6],
            "heap_raw_kb": row[7],
            "heap_compressed_kb": row[8]
        })

    return {"profiles": profiles, "count": len(profiles)}


@app.get("/api/profile/stats")
async def get_profile_stats():
    """Get aggregate profiling statistics."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            pgx_enabled,
            cpu_profiles_count,
            total_cpu_profile_raw_kb,
            total_cpu_profile_compressed_kb,
            avg_cpu_compression_ratio,
            heap_profiles_count,
            total_heap_profile_raw_kb,
            total_heap_profile_compressed_kb,
            avg_heap_compression_ratio,
            total_storage_kb,
            storage_per_query_kb,
            run_timestamp
        FROM aggregate_profiles
        ORDER BY run_timestamp DESC
    """)

    stats = []
    for row in cursor.fetchall():
        stats.append({
            "pgx_enabled": bool(row[0]),
            "cpu": {
                "count": row[1],
                "raw_kb": row[2],
                "compressed_kb": row[3],
                "compression_ratio": row[4]
            },
            "heap": {
                "count": row[5],
                "raw_kb": row[6],
                "compressed_kb": row[7],
                "compression_ratio": row[8]
            },
            "total_kb": row[9],
            "per_query_kb": row[10],
            "run_timestamp": row[11]
        })

    return {"stats": stats}


@app.get("/api/profile/cpu/{query_name}")
async def get_cpu_profile(
    query_name: str,
    run_timestamp: Optional[str] = None,
    pgx_enabled: bool = True
):
    """Get CPU profile data (decompressed) for a specific query."""
    conn = get_db()
    cursor = conn.cursor()

    # Get latest run if not specified
    if not run_timestamp or run_timestamp == "latest":
        cursor.execute("""
            SELECT p.run_timestamp
            FROM profiling p
            JOIN queries q ON p.query_id = q.query_id
            WHERE q.query_name = ? AND p.pgx_enabled = ?
            ORDER BY p.run_timestamp DESC LIMIT 1
        """, (query_name, pgx_enabled))
        row = cursor.fetchone()
        if row:
            run_timestamp = row[0]

    # Fetch profile data
    cursor.execute("""
        SELECT p.cpu_data_lz4, p.cpu_raw_size_kb
        FROM profiling p
        JOIN queries q ON p.query_id = q.query_id
        WHERE q.query_name = ?
          AND p.run_timestamp = ?
          AND p.pgx_enabled = ?
    """, (query_name, run_timestamp, pgx_enabled))

    row = cursor.fetchone()
    if not row or not row[0]:
        raise HTTPException(status_code=404, detail=f"CPU profile not found for {query_name}")

    # Decompress LZ4 data
    compressed_data = row[0]
    try:
        decompressed_data = lz4.frame.decompress(compressed_data)
        # Return raw perf.data file
        return Response(
            content=decompressed_data,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={query_name}_cpu.data"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decompress profile: {str(e)}")


# ============================================================================
# Static Files
# ============================================================================

# Serve static files (JS, CSS, images)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def read_root():
    """Serve index.html"""
    return FileResponse(STATIC_DIR / "index.html")


# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print(f"Starting benchmark dashboard server...")
    print(f"Database: {DB_PATH}")
    print(f"Static files: {STATIC_DIR}")
    print(f"")
    print(f"Access dashboard at:")
    print(f"  Local:   http://localhost:8000")
    print(f"  Network: http://<your-ip>:8000")
    print(f"")

    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces for LAN access
        port=8000,
        log_level="info"
    )
