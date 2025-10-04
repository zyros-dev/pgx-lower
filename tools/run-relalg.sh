#!/usr/bin/env bash
# run-relalg.sh - Execute MLIR RelAlg against TPC-H database
# Usage: ./tools/run-relalg.sh <column_types> <mlir_text>
#
# IMPORTANT: Use single quotes for MLIR text to preserve special characters like % and !
# Example: ./tools/run-relalg.sh 'c_custkey INTEGER' 'module { ... %0 ... }'

set -euo pipefail +H

if [ $# -lt 2 ]; then
    echo "Error: Missing required parameters"
    echo "Usage: $0 <column_types> <mlir_text>"
    exit 1
fi

COLUMN_TYPES="$1"
MLIR_TEXT="$2"

if ! sudo -u postgres /usr/local/pgsql/bin/pg_ctl status -D /usr/local/pgsql/data >/dev/null 2>&1; then
    echo "Error: PostgreSQL is not running"
    exit 1
fi

if [ ! -f /usr/local/pgsql/lib/pgx_lower.so ]; then
    echo "Warning: pgx_lower extension not found. Building..."
    ./tools/qbuild.sh ptest
fi

if [ ! -f tests/sql/init_tpch.sql ]; then
    echo "Error: tests/sql/init_tpch.sql not found"
    exit 1
fi

echo "Setting up TPC-H database..."
sudo -u postgres /usr/local/pgsql/bin/psql -d postgres -v ON_ERROR_STOP=1 << 'EOF'
DROP DATABASE IF EXISTS relalg_test;
CREATE DATABASE relalg_test;
EOF

echo "Loading TPC-H schema and data..."
sudo -u postgres /usr/local/pgsql/bin/psql -d relalg_test -v ON_ERROR_STOP=1 -f tests/sql/init_tpch.sql > /dev/null 2>&1

echo ""
echo "Fetching table OIDs and substituting in MLIR..."

MLIR_SUBSTITUTED="$MLIR_TEXT"

while IFS='|' read -r name oid; do
    name=$(echo "$name" | xargs)
    oid=$(echo "$oid" | xargs)
    [ -n "$name" ] && [ -n "$oid" ] || continue
    echo "  $name = $oid"

    MLIR_SUBSTITUTED=$(echo "$MLIR_SUBSTITUTED" | sed "s#table_identifier = \"$name|oid:[0-9]*\"#table_identifier = \"$name|oid:$oid\"#g")
done < <(sudo -u postgres /usr/local/pgsql/bin/psql -d relalg_test -t -F'|' << 'OIDSQL'
SELECT relname, oid
FROM pg_class
WHERE relname IN ('customer', 'orders', 'lineitem', 'part', 'partsupp', 'supplier', 'nation', 'region')
ORDER BY relname;
OIDSQL
)

echo ""
echo "Executing MLIR RelAlg..."
echo "NOTE: Returns only LAST ROW due to streaming model."
echo ""


TMPFILE="/tmp/run-relalg-$$.sql"
trap "rm -f $TMPFILE" EXIT

cat > "$TMPFILE" << SQLEND
CREATE EXTENSION IF NOT EXISTS pgx_lower;
SET client_min_messages TO WARNING;
SET pgx_lower.log_enable = false;
\\timing on

SELECT * FROM pgx_lower_test_relalg(\$\$
${MLIR_SUBSTITUTED}
\$\$) AS result(${COLUMN_TYPES});
SQLEND

chmod 644 "$TMPFILE"


sudo -u postgres /usr/local/pgsql/bin/psql -d relalg_test -v ON_ERROR_STOP=1 -f "$TMPFILE"

echo ""
echo "Execution completed!"
