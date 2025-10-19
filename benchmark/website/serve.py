#!/usr/bin/env python3

import json
import lz4.frame
import sqlite3
import base64
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

WEBSITE_DIR = Path(__file__).parent
OUTPUT_DIR = WEBSITE_DIR.parent / "output"
ARCHIVE_DIR = OUTPUT_DIR / "archive"
DEFAULT_DB_PATH = OUTPUT_DIR / "benchmark.db"
STATIC_DIR = WEBSITE_DIR / "static"

app = FastAPI(title="pgx-lower Benchmark Dashboard", version="2.0.0")

# Global to track current database (can be changed via API)
current_db_path = DEFAULT_DB_PATH

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db(db_name: str = None):
    """Get database connection. If db_name provided, use archive DB, otherwise use current."""
    global current_db_path

    if db_name:
        # Look for database in archive directory
        db_path = ARCHIVE_DIR / db_name
        if not db_path.exists():
            # Try adding .db extension if not present
            if not db_name.endswith('.db'):
                db_path = ARCHIVE_DIR / f"{db_name}.db"
        if not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Database not found: {db_name}")
    else:
        db_path = current_db_path
        if not db_path.exists():
            raise HTTPException(status_code=503, detail=f"Database not found: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/api/raw")
async def get_raw(
        run_timestamp: str = None,
        scale_factor: float = None,
        query_name: str = None,
        pgx_enabled: bool = None,
        include_flamegraph: bool = False,
        container: str = None,
        db: str = None
):
    conn = get_db(db)
    cursor = conn.cursor()

    conditions = []
    params = []

    if run_timestamp:
        conditions.append("r.run_timestamp = ?")
        params.append(run_timestamp)
    if scale_factor is not None:
        conditions.append("r.scale_factor = ?")
        params.append(scale_factor)
    if query_name:
        conditions.append("q.query_name = ?")
        params.append(query_name)
    if pgx_enabled is not None:
        conditions.append("q.pgx_enabled = ?")
        params.append(1 if pgx_enabled else 0)
    if container:
        conditions.append("r.container = ?")
        params.append(container)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    if include_flamegraph:
        # For flamegraphs, we need to join to profiling data by query_name and pgx_enabled
        # since profiling data comes from a different run (host-profile container)
        sql = f"""
            SELECT
                q.query_id,
                q.query_name,
                q.pgx_enabled,
                q.execution_metadata,
                q.postgres_metrics,
                q.metrics_json,
                q.result_validation,
                r.run_timestamp,
                r.scale_factor,
                r.pgx_version,
                r.postgres_version,
                prof.cpu_flamegraph_lz4,
                prof.heap_flamegraph_lz4
            FROM queries q
            JOIN runs r ON q.run_id = r.run_id
            LEFT JOIN (
                SELECT q2.query_name, q2.pgx_enabled, p.cpu_flamegraph_lz4, p.heap_flamegraph_lz4
                FROM queries q2
                JOIN profiling p ON q2.query_id = p.query_id
            ) prof ON q.query_name = prof.query_name AND q.pgx_enabled = prof.pgx_enabled
            WHERE {where_clause}
            ORDER BY r.run_timestamp DESC, q.query_name, q.pgx_enabled
        """
    else:
        sql = f"""
            SELECT
                q.query_id,
                q.query_name,
                q.pgx_enabled,
                q.execution_metadata,
                q.postgres_metrics,
                q.metrics_json,
                q.result_validation,
                r.run_timestamp,
                r.scale_factor,
                r.pgx_version,
                r.postgres_version,
                NULL as cpu_flamegraph_lz4,
                NULL as heap_flamegraph_lz4
            FROM queries q
            JOIN runs r ON q.run_id = r.run_id
            WHERE {where_clause}
            ORDER BY r.run_timestamp DESC, q.query_name, q.pgx_enabled
        """

    cursor.execute(sql, params)

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        # Parse execution_metadata
        exec_meta = json.loads(row[3]) if row[3] else {}

        # Parse postgres_metrics
        pg_metrics = json.loads(row[4]) if row[4] else {}
        buffers = pg_metrics.get("buffers", {})

        # Parse metrics_json (system-level metrics)
        sys_metrics = json.loads(row[5]) if row[5] else {}

        # Parse result_validation
        result_validation = json.loads(row[6]) if row[6] else {}
        result_hash = result_validation.get("hash", "")
        result_hash_short = result_hash[:8] if result_hash else None

        result = {
            "query_id": row[0],
            "query_name": row[1],
            "pgx_enabled": bool(row[2]),
            "status": exec_meta.get("status"),
            "duration_ms": round(exec_meta.get("duration_ms", 0), 2) if exec_meta.get("duration_ms") else None,
            "row_count": exec_meta.get("row_count"),
            "result_hash": result_hash_short,
            "exec_time": round(pg_metrics.get("execution_time_ms", 0), 2),
            "planning_time": round(pg_metrics.get("planning_time_ms", 0), 2),
            "shared_hit": buffers.get("shared_hit", 0),
            "shared_read": buffers.get("shared_read", 0),
            "shared_written": buffers.get("shared_written", 0),
            "temp_read": buffers.get("temp_read", 0),
            "temp_written": buffers.get("temp_written", 0),
            "spilled": pg_metrics.get("spilled_to_disk", False),
            "cpu_user_sec": sys_metrics.get("cpu_user_sec", 0),
            "cpu_system_sec": sys_metrics.get("cpu_system_sec", 0),
            "cpu_total_sec": sys_metrics.get("cpu_user_sec", 0) + sys_metrics.get("cpu_system_sec", 0),
            "cpu_percent": sys_metrics.get("cpu_percent", 0),
            "memory_peak_mb": sys_metrics.get("memory_peak_mb", 0),
            "io_read_mb": sys_metrics.get("io_read_mb", 0),
            "io_write_mb": sys_metrics.get("io_write_mb", 0),
            "run_timestamp": row[7],
            "scale_factor": row[8],
            "pgx_version": row[9],
            "postgres_version": row[10]
        }

        if include_flamegraph:
            if row[11]:  # cpu_flamegraph_lz4
                try:
                    result["cpu_flamegraph_lz4"] = base64.b64encode(row[11]).decode('ascii')
                except Exception as e:
                    print(f"CPU flamegraph encoding failed for {row[1]}: {e}")
                    result["cpu_flamegraph_lz4"] = None

            if row[12]:  # heap_flamegraph_lz4
                try:
                    result["heap_flamegraph_lz4"] = base64.b64encode(row[12]).decode('ascii')
                except Exception as e:
                    print(f"Heap flamegraph encoding failed for {row[1]}: {e}")
                    result["heap_flamegraph_lz4"] = None

        results.append(result)

    return results


@app.get("/api/aggregate")
async def get_aggregate(
        run_timestamp: str = None,
        scale_factor: float = None,
        container: str = None,
        db: str = None
):
    conn = get_db(db)
    cursor = conn.cursor()

    # Build WHERE clause
    conditions = []
    params = []

    if run_timestamp:
        conditions.append("r.run_timestamp = ?")
        params.append(run_timestamp)
    if scale_factor is not None:
        conditions.append("r.scale_factor = ?")
        params.append(scale_factor)
    if container:
        conditions.append("r.container = ?")
        params.append(container)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    cursor.execute(f"""
        SELECT
            r.run_timestamp,
            r.scale_factor,
            q.pgx_enabled,
            COUNT(*) as query_count,
            AVG(json_extract(q.postgres_metrics, '$.execution_time_ms')) as avg_exec_time,
            MIN(json_extract(q.postgres_metrics, '$.execution_time_ms')) as min_exec_time,
            MAX(json_extract(q.postgres_metrics, '$.execution_time_ms')) as max_exec_time,
            AVG(json_extract(q.postgres_metrics, '$.planning_time_ms')) as avg_plan_time,
            SUM(CASE WHEN json_extract(q.execution_metadata, '$.status') = 'SUCCESS' THEN 1 ELSE 0 END) as success_count,
            SUM(CASE WHEN json_extract(q.execution_metadata, '$.status') != 'SUCCESS' THEN 1 ELSE 0 END) as failure_count,
            r.pgx_version,
            r.postgres_version
        FROM queries q
        JOIN runs r ON q.run_id = r.run_id
        WHERE {where_clause}
        GROUP BY r.run_timestamp, r.scale_factor, q.pgx_enabled
        ORDER BY r.run_timestamp DESC, r.scale_factor, q.pgx_enabled DESC
    """, params)

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "run_timestamp": row[0],
            "scale_factor": row[1],
            "pgx_enabled": bool(row[2]),
            "query_count": row[3],
            "avg_exec_time": round(row[4], 2) if row[4] else None,
            "min_exec_time": round(row[5], 2) if row[5] else None,
            "max_exec_time": round(row[6], 2) if row[6] else None,
            "avg_plan_time": round(row[7], 2) if row[7] else None,
            "success_count": row[8],
            "failure_count": row[9],
            "pgx_version": row[10],
            "postgres_version": row[11]
        })

    return results


@app.get("/api/meta")
async def get_meta(db: str = None):
    """
    Get metadata about available runs and configurations.
    Tells you what data exists.
    """
    conn = get_db(db)
    cursor = conn.cursor()

    # Get all unique run timestamps and scale factors
    cursor.execute("""
                   SELECT r.run_timestamp,
                          r.scale_factor,
                          COUNT(DISTINCT q.query_name) as query_count,
                          r.pgx_version,
                          r.postgres_version
                   FROM runs r
                   LEFT JOIN queries q ON r.run_id = q.run_id
                   GROUP BY r.run_timestamp, r.scale_factor, r.pgx_version, r.postgres_version
                   ORDER BY r.run_timestamp DESC
                   """)

    runs = []
    for row in cursor.fetchall():
        runs.append({
            "run_timestamp": row[0],
            "scale_factor": row[1],
            "query_count": row[2],
            "pgx_version": row[3],
            "postgres_version": row[4]
        })

    # Get all unique query names
    cursor.execute("SELECT DISTINCT query_name FROM queries ORDER BY query_name")
    queries = [row[0] for row in cursor.fetchall()]

    # Get all unique scale factors from runs table
    cursor.execute("SELECT DISTINCT scale_factor FROM runs ORDER BY scale_factor")
    scale_factors = [row[0] for row in cursor.fetchall()]

    # Latest run
    cursor.execute("SELECT run_timestamp, scale_factor FROM runs ORDER BY run_timestamp DESC LIMIT 1")
    row = cursor.fetchone()
    latest = {"run_timestamp": row[0], "scale_factor": row[1]} if row else None

    conn.close()

    # List available databases
    databases = []

    # Current database
    if DEFAULT_DB_PATH.exists():
        stat = DEFAULT_DB_PATH.stat()
        databases.append({
            "name": "current",
            "path": "benchmark.db",
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_current": True
        })

    # Archived databases
    if ARCHIVE_DIR.exists():
        for db_file in sorted(ARCHIVE_DIR.glob("benchmark_*.db"), reverse=True):
            stat = db_file.stat()
            databases.append({
                "name": db_file.stem,  # Without .db extension
                "path": str(db_file.relative_to(OUTPUT_DIR)),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "is_current": False
            })

    return {
        "runs": runs,
        "queries": queries,
        "scale_factors": scale_factors,
        "latest": latest,
        "databases": databases
    }


# Static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def read_root():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    print(f"Starting SIMPLIFIED benchmark dashboard...")
    print(f"Database: {DEFAULT_DB_PATH}")
    print(f"Archive:  {ARCHIVE_DIR}")
    print(f"")
    print(f"API Endpoints:")
    print(f"  /api/meta      - Available runs/queries/scales/databases")
    print(f"  /api/raw       - Query executions (filterable)")
    print(f"  /api/aggregate - Grouped statistics (filterable)")
    print(f"")
    print(f"Access: http://localhost:8000")
    print(f"")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
