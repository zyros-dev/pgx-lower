#!/usr/bin/env python3
"""
TPC-H Benchmarking System for pgx-lower

Usage:
    python3 tools/bench.py pgx <scale_factor>      # Benchmark pgx-lower with JIT
    python3 tools/bench.py psql <scale_factor>     # Benchmark vanilla PostgreSQL (future)
    python3 tools/bench.py lingo <scale_factor>    # Benchmark LingoDB (future)
    python3 tools/bench.py validate <scale_factor> # Cross-validate all engines (future)

Example:
    python3 tools/bench.py pgx 0.1
    python3 tools/bench.py pgx 1
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
RESOURCES_SQL = PROJECT_ROOT / "resources" / "sql" / "tpch"
RESOURCES_DATA = PROJECT_ROOT / "resources" / "data" / "tpch"
RESULTS_DIR = PROJECT_ROOT / "benchmark_results"

# Database configuration
PGHOST = os.environ.get("PGHOST", "localhost")
PGPORT = os.environ.get("PGPORT", "5432")
PGUSER = os.environ.get("PGUSER", "postgres")
PGDATABASE = os.environ.get("PGDATABASE", "postgres")


def log(message):
    """Print timestamped log message"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def run_command(cmd, cwd=None, check=True, capture_output=False):
    """Run shell command with error handling"""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=capture_output, text=True)

    if check and result.returncode != 0:
        if capture_output and (result.stdout or result.stderr):
            print(result.stdout + result.stderr, file=sys.stderr)
        sys.exit(f"command failed: {cmd}")

    return result


def setup_tpch_data(scale_factor):
    """Download, compile dbgen, and generate TPC-H data"""
    log(f"Setting up TPC-H data (Scale Factor: {scale_factor})")

    # Create temporary directory
    tmpdir = tempfile.mkdtemp(prefix="tpch_")
    log(f"Working directory: {tmpdir}")

    try:
        # Download dbgen
        log("Downloading TPC-H dbgen...")
        run_command(
            "wget -q https://github.com/electrum/tpch-dbgen/archive/32f1c1b92d1664dba542e927d23d86ffa57aa253.zip -O tpch-dbgen.zip",
            cwd=tmpdir
        )

        # Extract
        run_command("unzip -q tpch-dbgen.zip", cwd=tmpdir)
        run_command(f"mv tpch-dbgen-32f1c1b92d1664dba542e927d23d86ffa57aa253/* .", cwd=tmpdir)
        run_command("rm tpch-dbgen.zip", cwd=tmpdir)

        # Compile dbgen (manual compilation to handle modern gcc)
        log("Compiling dbgen...")
        cflags = (
            "-g -DDBNAME=\\\"dss\\\" -DMAC -DORACLE -DTPCH -DRNG_TEST "
            "-D_FILE_OFFSET_BITS=64 -Wno-error=implicit-function-declaration "
            "-Wno-error=format-overflow"
        )

        sources = [
            "build.c", "driver.c", "bm_utils.c", "rnd.c", "print.c",
            "load_stub.c", "bcd2.c", "speed_seed.c", "text.c", "permute.c", "rng64.c"
        ]

        for source in sources:
            run_command(f"gcc {cflags} -c {source} 2>&1 | grep -v warning || true", cwd=tmpdir)

        run_command(
            "gcc build.o driver.o bm_utils.o rnd.o print.o load_stub.o bcd2.o "
            "speed_seed.o text.o permute.o rng64.o -o dbgen -lm",
            cwd=tmpdir
        )

        if not (Path(tmpdir) / "dbgen").exists():
            sys.exit("failed to compile dbgen")

        # Generate data
        log(f"Generating TPC-H data (this may take a while for large scale factors)...")
        run_command(f"./dbgen -f -s {scale_factor}", cwd=tmpdir)

        tbl_files = list(Path(tmpdir).glob("*.tbl"))
        if not tbl_files:
            sys.exit("no .tbl files generated")

        for tbl_file in tbl_files:
            # Fix permissions if needed
            tbl_file.chmod(0o644)
            # Remove trailing | from each line
            with open(tbl_file, 'r') as f:
                content = f.read()
            with open(tbl_file, 'w') as f:
                f.write(content.replace('|\n', '\n'))

        log(f"Generated {len(tbl_files)} table files")
        return tmpdir

    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise e


def create_database(db_name, scale_factor, data_dir):
    """Create PostgreSQL database and load TPC-H schema and data"""
    log(f"Creating database: {db_name}")

    # Drop and recreate database
    run_command(
        f"dropdb --if-exists -h {PGHOST} -p {PGPORT} -U {PGUSER} {db_name}",
        check=False
    )
    run_command(f"createdb -h {PGHOST} -p {PGPORT} -U {PGUSER} {db_name}")

    # Load schema
    log("Loading TPC-H schema...")
    schema_file = RESOURCES_SQL / "initialize.sql"
    run_command(f"psql -h {PGHOST} -p {PGPORT} -U {PGUSER} -d {db_name} -f {schema_file}")

    # Load data using COPY
    log("Loading data files...")
    for table in ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]:
        tbl_file = Path(data_dir) / f"{table}.tbl"
        if tbl_file.exists():
            run_command(
                f"psql -h {PGHOST} -p {PGPORT} -U {PGUSER} -d {db_name} -c \"\\\\COPY {table} FROM '{tbl_file}' WITH (FORMAT csv, DELIMITER '|')\"",
                check=False
            )

    # Create indexes
    log("Creating indexes...")
    index_sql = """
    CREATE INDEX IF NOT EXISTS idx_nation_regionkey ON nation(n_regionkey);
    CREATE INDEX IF NOT EXISTS idx_supplier_nationkey ON supplier(s_nationkey);
    CREATE INDEX IF NOT EXISTS idx_partsupp_partkey ON partsupp(ps_partkey);
    CREATE INDEX IF NOT EXISTS idx_partsupp_suppkey ON partsupp(ps_suppkey);
    CREATE INDEX IF NOT EXISTS idx_customer_nationkey ON customer(c_nationkey);
    CREATE INDEX IF NOT EXISTS idx_orders_custkey ON orders(o_custkey);
    CREATE INDEX IF NOT EXISTS idx_lineitem_orderkey ON lineitem(l_orderkey);
    CREATE INDEX IF NOT EXISTS idx_lineitem_partkey ON lineitem(l_partkey);
    CREATE INDEX IF NOT EXISTS idx_lineitem_suppkey ON lineitem(l_suppkey);

    ANALYZE region;
    ANALYZE nation;
    ANALYZE part;
    ANALYZE supplier;
    ANALYZE partsupp;
    ANALYZE customer;
    ANALYZE orders;
    ANALYZE lineitem;
    """
    run_command(f"psql -h {PGHOST} -p {PGPORT} -U {PGUSER} -d {db_name} -c \"{index_sql}\"")

    # Show database statistics
    stats_sql = """
    SELECT
        schemaname,
        relname AS tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) AS size,
        n_live_tup AS rows
    FROM pg_stat_user_tables
    ORDER BY pg_total_relation_size(schemaname||'.'||relname) DESC;
    """
    result = run_command(
        f"psql -h {PGHOST} -p {PGPORT} -U {PGUSER} -d {db_name} -c \"{stats_sql}\"",
        capture_output=True
    )
    log("Database statistics:")
    print(result.stdout)


def run_pgx_benchmark(scale_factor):
    """Run TPC-H benchmark using pgx-lower JIT compilation"""
    log("=" * 60)
    log(f"TPC-H Benchmark - pgx-lower (Scale Factor: {scale_factor})")
    log("=" * 60)

    # Setup data
    data_dir = setup_tpch_data(scale_factor)

    # Create database
    sf_str = str(scale_factor).replace(".", "_")
    db_name = f"tpch_sf{sf_str}"

    try:
        create_database(db_name, scale_factor, data_dir)

        # Create results directory
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"tpch_sf{sf_str}_{timestamp}.txt"

        log(f"Running TPC-H queries (results: {result_file})")

        with open(result_file, "w") as f:
            # Write header
            f.write("=" * 60 + "\n")
            f.write("TPC-H Benchmark Results - pgx-lower\n")
            f.write(f"Scale Factor: {scale_factor}\n")
            f.write(f"Database: {db_name}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

            # Run each query
            for query_num in range(1, 23):
                log(f"Running Query {query_num}/22...")
                query_file = RESOURCES_SQL / f"{query_num}.sql"

                if not query_file.exists():
                    f.write(f"ERROR: Query file {query_file} not found\n\n")
                    continue

                # Prepare SQL with pgx-lower setup
                with open(query_file, "r") as qf:
                    query_sql = qf.read()

                full_sql = (
                    "LOAD 'pgx_lower.so';\n"
                    "SET client_min_messages TO NOTICE;\n"
                    "SET pgx_lower.log_enable = true;\n\n"
                    f"{query_sql}"
                )

                # Write query header to results
                f.write("=" * 60 + "\n")
                f.write(f"TPC-H Query {query_num}\n")
                f.write("=" * 60 + "\n")
                f.write(query_sql.strip() + "\n\n")
                f.write("Results:\n")
                f.write("-" * 60 + "\n")

                # Execute query
                result = run_command(
                    f"psql -h {PGHOST} -p {PGPORT} -U {PGUSER} -d {db_name} -c \"{full_sql}\"",
                    capture_output=True,
                    check=False
                )

                # Write results
                f.write(result.stdout)
                if result.stderr:
                    f.write("\nSTDERR:\n")
                    f.write(result.stderr)
                f.write("\n\n")

                if result.returncode != 0:
                    f.write(f"ERROR: Query {query_num} failed with return code {result.returncode}\n\n")

        log("=" * 60)
        log("Benchmark Complete!")
        log("=" * 60)
        log(f"Results saved to: {result_file}")
        log(f"Database: {db_name} (preserved for analysis)")
        log(f"\nTo view results: cat {result_file}")
        log(f"To clean up: dropdb -h {PGHOST} -p {PGPORT} -U {PGUSER} {db_name}")

    finally:
        # Cleanup temporary data directory
        shutil.rmtree(data_dir, ignore_errors=True)


def run_psql_benchmark(scale_factor):
    """Run TPC-H benchmark using vanilla PostgreSQL (no JIT)"""
    sys.exit("not implemented")


def run_lingo_benchmark(scale_factor):
    """Run TPC-H benchmark using LingoDB"""
    sys.exit("not implemented")


def run_validate_benchmark(scale_factor):
    """Cross-validate results from all benchmark engines"""
    sys.exit("not implemented")


def main():
    parser = argparse.ArgumentParser(
        description="TPC-H Benchmarking System for pgx-lower",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python3 tools/bench.py pgx 0.1      # small test dataset
  python3 tools/bench.py pgx 1        # standard 1GB dataset
  python3 tools/bench.py pgx 10       # large 10GB dataset"""
    )

    parser.add_argument(
        "mode",
        choices=["pgx", "psql", "lingo", "validate"],
        help="Benchmark mode: pgx (pgx-lower), psql (vanilla), lingo (LingoDB), validate (cross-check)"
    )

    parser.add_argument(
        "scale_factor",
        type=float,
        help="TPC-H scale factor (e.g., 0.1, 1, 10)"
    )

    args = parser.parse_args()

    if args.scale_factor <= 0:
        sys.exit("scale factor must be positive")

    # Dispatch to appropriate benchmark function
    if args.mode == "pgx":
        run_pgx_benchmark(args.scale_factor)
    elif args.mode == "psql":
        run_psql_benchmark(args.scale_factor)
    elif args.mode == "lingo":
        run_lingo_benchmark(args.scale_factor)
    elif args.mode == "validate":
        run_validate_benchmark(args.scale_factor)


if __name__ == "__main__":
    main()
