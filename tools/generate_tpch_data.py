#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

DBGEN_REPO = "https://github.com/electrum/tpch-dbgen/archive/32f1c1b92d1664dba542e927d23d86ffa57aa253.zip"
DBGEN_ZIP = "tpch-dbgen.zip"

TABLE_ORDER = ['region', 'nation', 'part', 'supplier', 'partsupp', 'customer', 'orders', 'lineitem']

SCHEMA = {
    'region': {
        'columns': [
            ('r_regionkey', 'INTEGER', 'NOT NULL'),
            ('r_name', 'CHAR(25)', 'NOT NULL'),
            ('r_comment', 'VARCHAR(152)', 'NOT NULL'),
        ],
        'pkey': ['r_regionkey']
    },
    'nation': {
        'columns': [
            ('n_nationkey', 'INTEGER', 'NOT NULL'),
            ('n_name', 'CHAR(25)', 'NOT NULL'),
            ('n_regionkey', 'INTEGER', 'NOT NULL'),
            ('n_comment', 'VARCHAR(152)', 'NOT NULL'),
        ],
        'pkey': ['n_nationkey']
    },
    'part': {
        'columns': [
            ('p_partkey', 'INTEGER', 'NOT NULL'),
            ('p_name', 'VARCHAR(55)', 'NOT NULL'),
            ('p_mfgr', 'CHAR(25)', 'NOT NULL'),
            ('p_brand', 'CHAR(10)', 'NOT NULL'),
            ('p_type', 'VARCHAR(25)', 'NOT NULL'),
            ('p_size', 'INTEGER', 'NOT NULL'),
            ('p_container', 'CHAR(10)', 'NOT NULL'),
            ('p_retailprice', 'DECIMAL(12, 2)', 'NOT NULL'),
            ('p_comment', 'VARCHAR(23)', 'NOT NULL'),
        ],
        'pkey': ['p_partkey']
    },
    'supplier': {
        'columns': [
            ('s_suppkey', 'INTEGER', 'NOT NULL'),
            ('s_name', 'CHAR(25)', 'NOT NULL'),
            ('s_address', 'VARCHAR(40)', 'NOT NULL'),
            ('s_nationkey', 'INTEGER', 'NOT NULL'),
            ('s_phone', 'CHAR(15)', 'NOT NULL'),
            ('s_acctbal', 'DECIMAL(12, 2)', 'NOT NULL'),
            ('s_comment', 'VARCHAR(101)', 'NOT NULL'),
        ],
        'pkey': ['s_suppkey']
    },
    'partsupp': {
        'columns': [
            ('ps_partkey', 'INTEGER', 'NOT NULL'),
            ('ps_suppkey', 'INTEGER', 'NOT NULL'),
            ('ps_availqty', 'INTEGER', 'NOT NULL'),
            ('ps_supplycost', 'DECIMAL(12, 2)', 'NOT NULL'),
            ('ps_comment', 'VARCHAR(199)', 'NOT NULL'),
        ],
        'pkey': ['ps_partkey', 'ps_suppkey']
    },
    'customer': {
        'columns': [
            ('c_custkey', 'INTEGER', 'NOT NULL'),
            ('c_name', 'VARCHAR(25)', 'NOT NULL'),
            ('c_address', 'VARCHAR(40)', 'NOT NULL'),
            ('c_nationkey', 'INTEGER', 'NOT NULL'),
            ('c_phone', 'CHAR(15)', 'NOT NULL'),
            ('c_acctbal', 'DECIMAL(12, 2)', 'NOT NULL'),
            ('c_mktsegment', 'CHAR(10)', 'NOT NULL'),
            ('c_comment', 'VARCHAR(117)', 'NOT NULL'),
        ],
        'pkey': ['c_custkey']
    },
    'orders': {
        'columns': [
            ('o_orderkey', 'INTEGER', 'NOT NULL'),
            ('o_custkey', 'INTEGER', 'NOT NULL'),
            ('o_orderstatus', 'CHAR(1)', 'NOT NULL'),
            ('o_totalprice', 'DECIMAL(12, 2)', 'NOT NULL'),
            ('o_orderdate', 'DATE', 'NOT NULL'),
            ('o_orderpriority', 'CHAR(15)', 'NOT NULL'),
            ('o_clerk', 'CHAR(15)', 'NOT NULL'),
            ('o_shippriority', 'INTEGER', 'NOT NULL'),
            ('o_comment', 'VARCHAR(79)', 'NOT NULL'),
        ],
        'pkey': ['o_orderkey']
    },
    'lineitem': {
        'columns': [
            ('l_orderkey', 'INTEGER', 'NOT NULL'),
            ('l_partkey', 'INTEGER', 'NOT NULL'),
            ('l_suppkey', 'INTEGER', 'NOT NULL'),
            ('l_linenumber', 'INTEGER', 'NOT NULL'),
            ('l_quantity', 'DECIMAL(12, 2)', 'NOT NULL'),
            ('l_extendedprice', 'DECIMAL(12, 2)', 'NOT NULL'),
            ('l_discount', 'DECIMAL(12, 2)', 'NOT NULL'),
            ('l_tax', 'DECIMAL(12, 2)', 'NOT NULL'),
            ('l_returnflag', 'CHAR(1)', 'NOT NULL'),
            ('l_linestatus', 'CHAR(1)', 'NOT NULL'),
            ('l_shipdate', 'DATE', 'NOT NULL'),
            ('l_commitdate', 'DATE', 'NOT NULL'),
            ('l_receiptdate', 'DATE', 'NOT NULL'),
            ('l_shipinstruct', 'CHAR(25)', 'NOT NULL'),
            ('l_shipmode', 'CHAR(10)', 'NOT NULL'),
            ('l_comment', 'VARCHAR(44)', 'NOT NULL'),
        ],
        'pkey': ['l_orderkey', 'l_linenumber']
    },
}


def download_and_build_dbgen(tmpdir):
    print(f"Downloading tpch-dbgen to {tmpdir}")
    subprocess.run(['wget', '-q', DBGEN_REPO, '-O', DBGEN_ZIP], cwd=tmpdir, check=True)

    print("Extracting tpch-dbgen")
    subprocess.run(['unzip', '-q', DBGEN_ZIP], cwd=tmpdir, check=True)

    extracted = tmpdir / 'tpch-dbgen-32f1c1b92d1664dba542e927d23d86ffa57aa253'
    for item in extracted.iterdir():
        shutil.move(str(item), str(tmpdir))
    extracted.rmdir()
    (tmpdir / DBGEN_ZIP).unlink()

    print("Patching bm_utils.c for strdup declaration")
    bm_utils_path = tmpdir / 'bm_utils.c'
    with open(bm_utils_path, 'r') as f:
        content = f.read()

    if '#include <string.h>' in content and 'char *strdup' not in content:
        content = content.replace('#include <string.h>', '#include <string.h>\nchar *strdup(const char *s);')
        with open(bm_utils_path, 'w') as f:
            f.write(content)

    print("Building dbgen")
    result = subprocess.run(['make'], cwd=tmpdir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed with return code {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, ['make'], result.stdout, result.stderr)


def generate_data(tmpdir, scale_factor):
    print(f"Generating TPC-H data at scale factor {scale_factor}")
    subprocess.run(['./dbgen', '-f', '-s', str(scale_factor)], cwd=tmpdir, check=True)

    for tbl_file in tmpdir.glob('*.tbl'):
        tbl_file.chmod(0o644)
        with open(tbl_file, 'r') as f:
            content = f.read()
        content = content.replace('|\n', '\n')
        with open(tbl_file, 'w') as f:
            f.write(content)


def parse_tbl_file(tbl_path):
    rows = []
    with open(tbl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split('|')
            rows.append(fields)
    return rows


def format_value(value, col_type):
    value = value.strip()
    if col_type.startswith('INTEGER') or col_type.startswith('DECIMAL'):
        return value
    elif col_type.startswith('DATE'):
        return f"'{value}'"
    else:
        return f"'{value}'"


def generate_insert_statements(table_name, rows, schema, chunk_size=5000):
    if not rows:
        return ""

    col_names = [col[0] for col in schema['columns']]
    col_types = [col[1] for col in schema['columns']]

    chunks = []
    for chunk_start in range(0, len(rows), chunk_size):
        chunk_rows = rows[chunk_start:chunk_start + chunk_size]

        lines = [f"INSERT INTO {table_name}({', '.join(col_names)})"]
        lines.append("VALUES ")

        value_lines = []
        for row in chunk_rows:
            formatted_values = [format_value(row[i], col_types[i]) for i in range(len(col_names))]
            value_lines.append(f"       ({', '.join(formatted_values)})")

        lines.append(',\n'.join(value_lines) + ';')
        chunks.append('\n'.join(lines))

    return '\n'.join(chunks)


def generate_sql(tmpdir, output_path):
    print(f"Generating SQL file: {output_path}")

    with open(output_path, 'w') as out:
        out.write("DROP TABLE IF EXISTS part CASCADE;\n")
        out.write("DROP TABLE IF EXISTS region CASCADE;\n")
        out.write("DROP TABLE IF EXISTS nation CASCADE;\n")
        out.write("DROP TABLE IF EXISTS supplier CASCADE;\n")
        out.write("DROP TABLE IF EXISTS partsupp CASCADE;\n")
        out.write("DROP TABLE IF EXISTS customer CASCADE;\n")
        out.write("DROP TABLE IF EXISTS orders CASCADE;\n")
        out.write("DROP TABLE IF EXISTS lineitem CASCADE;\n")
        out.write("\n")

        for table_name in TABLE_ORDER:
            schema = SCHEMA[table_name]

            out.write(f"CREATE TABLE {table_name} (\n")
            col_defs = []
            for col_name, col_type, constraint in schema['columns']:
                col_defs.append(f"                          {col_name:15} {col_type:14} {constraint}")
            out.write(',\n'.join(col_defs))
            out.write(',\n')
            pkey_cols = ', '.join(schema['pkey'])
            out.write(f"                          PRIMARY KEY ({pkey_cols})\n")
            out.write(");\n\n")

        for table_name in TABLE_ORDER:
            tbl_file = tmpdir / f"{table_name}.tbl"
            if not tbl_file.exists():
                print(f"Warning: {tbl_file} not found, skipping")
                continue

            print(f"Processing {table_name}.tbl")
            rows = parse_tbl_file(tbl_file)
            insert_sql = generate_insert_statements(table_name, rows, SCHEMA[table_name])
            out.write(insert_sql)
            out.write("\n\n")

    print(f"Generated {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate TPC-H test data for pgx-lower')
    parser.add_argument('scale_factor', type=float, help='TPC-H scale factor (e.g., 0.01 for ~10MB)')
    parser.add_argument('--output', '-o', default='tests/sql/init_tpch.sql', help='Output SQL file path')
    args = parser.parse_args()

    tmpdir = Path(tempfile.mkdtemp())
    try:
        download_and_build_dbgen(tmpdir)
        generate_data(tmpdir, args.scale_factor)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generate_sql(tmpdir, output_path)

        print(f"\nDone! Generated TPC-H data at scale factor {args.scale_factor}")
        print(f"Output: {output_path}")

    finally:
        print(f"Cleaning up {tmpdir}")
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
