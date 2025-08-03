#!/bin/bash
#
# Queue Status - Show current build queue status
#

LOCK_FILE="/home/xzel/repos/pgx-lower/.build_lock"
QUEUE_DIR="/tmp/pgx_lower_queue"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== PGX-Lower Build Queue Status ===${NC}"
echo ""

# Check if build is currently running
if [[ -f "$LOCK_FILE" ]]; then
    lock_info=$(cat "$LOCK_FILE")
    lock_agent=$(echo "$lock_info" | cut -d'|' -f1)
    lock_pid=$(echo "$lock_info" | cut -d'|' -f2)
    lock_time=$(echo "$lock_info" | cut -d'|' -f3)
    lock_age=$(($(date +%s) - lock_time))
    
    if kill -0 "$lock_pid" 2>/dev/null; then
        echo -e "${GREEN}🔨 Currently Building:${NC}"
        echo "  Agent: $lock_agent"
        echo "  PID: $lock_pid"
        echo "  Duration: ${lock_age}s"
    else
        echo -e "${RED}⚠️  Stale Lock Detected:${NC}"
        echo "  Agent: $lock_agent (process died)"
        echo "  PID: $lock_pid (dead)"
        echo "  Age: ${lock_age}s"
        echo "  Recommendation: Lock will be cleaned up automatically"
    fi
else
    echo -e "${GREEN}✅ Build System Available${NC}"
fi

echo ""

# Check queue
if [[ -d "$QUEUE_DIR" ]]; then
    queue_files=($(ls -1 "$QUEUE_DIR"/queue_* 2>/dev/null | sort))
    if [[ ${#queue_files[@]} -gt 0 ]]; then
        echo -e "${YELLOW}📋 Agents in Queue (${#queue_files[@]}):${NC}"
        position=1
        for qfile in "${queue_files[@]}"; do
            if [[ -f "$qfile" ]]; then
                info=$(cat "$qfile")
                agent=$(echo "$info" | cut -d'|' -f1)
                cmd=$(echo "$info" | cut -d'|' -f2)
                timestamp=$(echo "$info" | cut -d'|' -f3)
                wait_time=$(($(date +%s) - timestamp))
                
                echo "  $position. $agent: $cmd (waiting ${wait_time}s)"
                ((position++))
            fi
        done
    else
        echo -e "${GREEN}📋 Queue Empty${NC}"
    fi
else
    echo -e "${GREEN}📋 Queue Empty${NC}"
fi

echo ""
echo -e "${BLUE}Available Commands:${NC}"
echo "  ./tools/qbuild.sh <operation>  - Queued build operations"
echo "  ./tools/qmake.sh <args>        - Queued make"
echo "  ./tools/qctest.sh <args>       - Queued unit tests"
echo "  ./tools/qstatus.sh             - This status display"