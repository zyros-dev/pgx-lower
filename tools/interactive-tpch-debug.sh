#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

echo -e "${BLUE}Step 1: Loading TPC-H data...${NC}"
echo "" >> "$DEBUG_LOG"
echo "==================== LOADING TPCH DATA ====================" >> "$DEBUG_LOG"
echo "" >> "$DEBUG_LOG"
$PSQL -f tests/sql/init_tpch.sql >> "$DEBUG_LOG" 2>&1

echo -e "${BLUE}Step 2: Starting monitored session...${NC}"
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
SET pgx_lower.enabled_categories = 'AST_TRANSLATE,RELALG_LOWER,DB_LOWER,DSA_LOWER,UTIL_LOWER';
SELECT pg_sleep($SLEEP_TIME);
$SQL_CONTENT
EOF

echo -e "${YELLOW}Sleeping ${SLEEP_TIME}s for GDB attachment${NC}"

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
    echo -e "${RED}Failed to get backend PID${NC}"
    kill $PSQL_PID 2>/dev/null || true
    exit 1
fi

echo -e "${GREEN}Backend PID: $BACKEND_PID${NC}"
echo "" >> "$DEBUG_LOG"
echo "=== GDB SESSION ===" >> "$DEBUG_LOG"
echo "Backend PID: $BACKEND_PID" >> "$DEBUG_LOG"
echo "" >> "$DEBUG_LOG"

GDB_COMMANDS=$(mktemp)
cat > "$GDB_COMMANDS" <<'GDBEOF'
handle SIGSEGV stop print
handle SIGABRT stop print
handle SIGUSR1 nostop noprint
set pagination off
set print pretty on

printf "\n=== GDB ATTACHED ===\n"
info threads
printf "\n=== CONTINUING EXECUTION (will break on SIGSEGV) ===\n"
continue

printf "\n=== CRASH DETECTED ===\n"
printf "You are now in interactive GDB. The crash has occurred.\n"
printf "Useful commands:\n"
printf "  bt        - full backtrace\n"
printf "  bt full   - backtrace with local variables\n"
printf "  frame N   - switch to frame N\n"
printf "  info locals - show local variables in current frame\n"
printf "  info args   - show function arguments\n"
printf "  p variable  - print variable value\n"
printf "  x/20i $rip  - examine assembly around instruction pointer\n"
printf "  info registers - show all registers\n"
printf "  quit      - exit GDB\n"
printf "\n"
backtrace
GDBEOF

echo -e "${BLUE}Attaching GDB to PID $BACKEND_PID (interactive mode)...${NC}"
echo -e "${YELLOW}GDB will stop when the crash occurs. You'll get an interactive prompt.${NC}"
echo ""

# Run GDB interactively (remove -batch flag)
sudo gdb -p "$BACKEND_PID" -x "$GDB_COMMANDS"

# Clean up after GDB exits
rm -f "$SQL_WRAPPER" "$GDB_COMMANDS"
kill $PSQL_PID 2>/dev/null || true

echo ""
echo -e "${GREEN}GDB session ended${NC}"
echo -e "${GREEN}Debug log: $DEBUG_LOG${NC}"
