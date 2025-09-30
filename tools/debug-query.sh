#!/bin/bash
set -e

PSQL="psql -h localhost -d postgres"
DEBUG_LOG="/tmp/pgx_debug.log"
SLEEP_TIME=30

if [ $# -eq 0 ]; then
    echo -e "Error: No SQL file or query provided"
    echo "Usage: $0 <sql_file_or_query>"
    exit 1
fi

INPUT="$1"

rm -f "$DEBUG_LOG"
echo "=== PGX-LOWER DEBUG SESSION ===" >> "$DEBUG_LOG"
echo "Timestamp: $(date)" >> "$DEBUG_LOG"
echo "" >> "$DEBUG_LOG"

if [ -f "$INPUT" ]; then
    echo -e "Using SQL file: $INPUT"
    echo "Input: SQL file $INPUT" >> "$DEBUG_LOG"
    SQL_CONTENT=$(cat "$INPUT")
else
    echo -e "Using SQL query: $INPUT"
    echo "Input: Direct SQL query" >> "$DEBUG_LOG"
    SQL_CONTENT="$INPUT"
fi

echo "" >> "$DEBUG_LOG"
echo "=== SQL TO EXECUTE ===" >> "$DEBUG_LOG"
echo "$SQL_CONTENT" >> "$DEBUG_LOG"
echo "" >> "$DEBUG_LOG"

SQL_WRAPPER=$(mktemp)
cat > "$SQL_WRAPPER" <<EOF
SELECT 'GDB_ATTACH_PID:' || pg_backend_pid();
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
