#!/bin/bash
set -e

PSQL="psql -h localhost -d postgres"
DEBUG_LOG="/tmp/pgx_tpch_debug.log"
SLEEP_TIME=30

SQL_CONTENT="select
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice,
    sum(l_quantity)
from
    customer,
    orders,
    lineitem
where
    o_orderkey in (
        select
            l_orderkey
        from
            lineitem
        group by
            l_orderkey having
                sum(l_quantity) > 300
    )
    and c_custkey = o_custkey
    and o_orderkey = l_orderkey
group by
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice
order by
    o_totalprice desc,
    o_orderdate
limit 100;"

rm -f "$DEBUG_LOG"
echo "=== PGX-LOWER TPCH DEBUG SESSION ===" >> "$DEBUG_LOG"
echo "Timestamp: $(date)" >> "$DEBUG_LOG"
echo "" >> "$DEBUG_LOG"

echo "" >> "$DEBUG_LOG"
echo "=== SQL TO EXECUTE ===" >> "$DEBUG_LOG"
echo "$SQL_CONTENT" >> "$DEBUG_LOG"
echo "" >> "$DEBUG_LOG"

echo -e "Step 1: Loading TPC-H data..."
echo "" >> "$DEBUG_LOG"
echo "==================== LOADING TPCH DATA ====================" >> "$DEBUG_LOG"
echo "" >> "$DEBUG_LOG"
$PSQL -f tests/sql/init_tpch.sql >> "$DEBUG_LOG" 2>&1

echo -e "Step 2: Starting monitored session..."
echo "" >> "$DEBUG_LOG"
echo "==================== STARTING QUERY SESSION ====================" >> "$DEBUG_LOG"
echo "" >> "$DEBUG_LOG"

SQL_WRAPPER=$(mktemp)
cat > "$SQL_WRAPPER" <<EOF
SELECT 'GDB_ATTACH_PID:' || pg_backend_pid();
LOAD 'pgx_lower';
SET client_min_messages TO DEBUG1;
SET pgx_lower.log_enable = true;
SET pgx_lower.log_debug = true;
SET pgx_lower.log_io = true;
SET pgx_lower.log_ir = true;
SET pgx_lower.log_trace = true;
SET pgx_lower.log_verbose = true;
SET pgx_lower.enabled_categories = 'AST_TRANSLATE,RELALG_LOWER,DB_LOWER,DSA_LOWER,UTIL_LOWER,JIT,GENERAL';
SELECT pg_sleep($SLEEP_TIME);
$SQL_CONTENT
EOF

echo -e "Sleeping ${SLEEP_TIME}s for GDB attachment"

$PSQL -f "$SQL_WRAPPER" >> "$DEBUG_LOG" 2>&1 &
PSQL_PID=$!

sleep 2

BACKEND_PID=""
for i in {1..10}; do
    BACKEND_PID=$(grep "GDB_ATTACH_PID:" "$DEBUG_LOG" 2>/dev/null | tail -1 | sed 's/.*GDB_ATTACH_PID://' | tr -d ' ' || true)
    if [ -n "$BACKEND_PID" ]; then
        break
    fi
    sleep 1
done

if [ -z "$BACKEND_PID" ]; then
    echo -e "Failed to get backend PID"
    kill $PSQL_PID 2>/dev/null || true
    exit 1
fi

echo -e "Backend PID: $BACKEND_PID"
echo "" >> "$DEBUG_LOG"
echo "=== GDB SESSION ===" >> "$DEBUG_LOG"
echo "Backend PID: $BACKEND_PID" >> "$DEBUG_LOG"
echo "" >> "$DEBUG_LOG"

GDB_COMMANDS=$(mktemp)
cat > "$GDB_COMMANDS" <<'GDBEOF'
handle SIGSEGV stop print
handle SIGABRT stop print
set pagination off
set print pretty on

printf "\n=== GDB ATTACHED ===\n"
info threads
printf "\n=== CONTINUING EXECUTION ===\n"
continue

printf "\n=== CRASH DETECTED ===\n"
backtrace full

printf "\n=== FRAME 0 DETAILS ===\n"
frame 0
info locals
info args
info registers

printf "\n=== FRAME 1 DETAILS ===\n"
frame 1
info locals
info args

printf "\n=== FRAME 2 DETAILS ===\n"
frame 2
info locals
info args

printf "\n=== ASSEMBLY CONTEXT ===\n"
x/20i $rip-40

printf "\n=== MEMORY CONTEXT ===\n"
x/64xb $rsp

quit
GDBEOF

echo -e "Attaching GDB to PID $BACKEND_PID..."
sudo gdb -batch -p "$BACKEND_PID" -x "$GDB_COMMANDS" >> "$DEBUG_LOG" 2>&1 &
GDB_PID=$!

echo -e "Waiting for query execution..."
wait $PSQL_PID 2>/dev/null || true
wait $GDB_PID 2>/dev/null || true

rm -f "$SQL_WRAPPER" "$GDB_COMMANDS"

echo ""
echo -e "Execution complete"
echo -e "Debug log: $DEBUG_LOG"
echo ""

if grep -q "CRASH DETECTED" "$DEBUG_LOG" 2>/dev/null; then
    echo -e "Crash detected - see $DEBUG_LOG for details"
    exit 1
else
    echo -e "No crash detected"
    exit 0
fi
