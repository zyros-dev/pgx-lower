#!/bin/bash
#
# Build Queue System for pgx-lower Multi-Agent Development
# Serializes build operations to prevent concurrent compilation conflicts
#

set -e

# Configuration
LOCK_FILE="/home/xzel/repos/pgx-lower/.build_lock"
QUEUE_DIR="/tmp/pgx_lower_queue"
MAX_WAIT_TIME=1800  # 30 minutes max wait
BUILD_TIMEOUT=900   # 15 minutes max per build
POLL_INTERVAL=2     # Check every 2 seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create queue directory
mkdir -p "$QUEUE_DIR"

# Function to clean up on exit
cleanup() {
    if [[ -n "$QUEUE_ID" && -f "$QUEUE_DIR/$QUEUE_ID" ]]; then
        rm -f "$QUEUE_DIR/$QUEUE_ID"
    fi
    if [[ "$LOCK_ACQUIRED" == "true" ]]; then
        rm -f "$LOCK_FILE"
        echo -e "${GREEN}[BUILD QUEUE] Released build lock${NC}"
    fi
}

# Set up cleanup on script exit
trap cleanup EXIT INT TERM

# Generate unique queue ID
QUEUE_ID="queue_$$_$(date +%s)"
LOCK_ACQUIRED="false"

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <build_command> [args...]"
    echo "Examples:"
    echo "  $0 make -j4"
    echo "  $0 cmake --build ."
    echo "  $0 ninja"
    echo "  $0 ctest"
    exit 1
fi

BUILD_COMMAND="$*"
AGENT_NAME="${CLAUDE_AGENT_NAME:-unknown-agent}"

echo -e "${BLUE}[BUILD QUEUE] Agent '$AGENT_NAME' requesting build: $BUILD_COMMAND${NC}"

# Add ourselves to the queue
echo "$AGENT_NAME|$BUILD_COMMAND|$(date +%s)" > "$QUEUE_DIR/$QUEUE_ID"

# Function to show queue status
show_queue_status() {
    local queue_files=($(ls -1 "$QUEUE_DIR"/queue_* 2>/dev/null | sort))
    if [[ ${#queue_files[@]} -gt 1 ]]; then
        echo -e "${YELLOW}[BUILD QUEUE] Current queue (${#queue_files[@]} agents waiting):${NC}"
        local position=1
        for qfile in "${queue_files[@]}"; do
            if [[ -f "$qfile" ]]; then
                local info=$(cat "$qfile")
                local agent=$(echo "$info" | cut -d'|' -f1)
                local cmd=$(echo "$info" | cut -d'|' -f2)
                local timestamp=$(echo "$info" | cut -d'|' -f3)
                local wait_time=$(($(date +%s) - timestamp))
                
                if [[ "$(basename "$qfile")" == "$QUEUE_ID" ]]; then
                    echo -e "  ${GREEN}→ $position. $agent: $cmd (waiting ${wait_time}s)${NC}"
                else
                    echo -e "    $position. $agent: $cmd (waiting ${wait_time}s)"
                fi
                ((position++))
            fi
        done
    fi
}

# Wait for our turn
wait_start=$(date +%s)
while true; do
    current_time=$(date +%s)
    wait_duration=$((current_time - wait_start))
    
    # Check if we've waited too long
    if [[ $wait_duration -gt $MAX_WAIT_TIME ]]; then
        echo -e "${RED}[BUILD QUEUE] Timeout: waited $wait_duration seconds, giving up${NC}"
        exit 1
    fi
    
    # Try to acquire the lock
    if (set -C; echo "$AGENT_NAME|$$|$(date +%s)" > "$LOCK_FILE") 2>/dev/null; then
        LOCK_ACQUIRED="true"
        echo -e "${GREEN}[BUILD QUEUE] Lock acquired by agent '$AGENT_NAME'${NC}"
        break
    fi
    
    # Check if lock is stale (holder process died)
    if [[ -f "$LOCK_FILE" ]]; then
        lock_info=$(cat "$LOCK_FILE")
        lock_pid=$(echo "$lock_info" | cut -d'|' -f2)
        lock_time=$(echo "$lock_info" | cut -d'|' -f3)
        lock_age=$((current_time - lock_time))
        
        # If lock is older than BUILD_TIMEOUT and process is dead, clean it up
        if [[ $lock_age -gt $BUILD_TIMEOUT ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
            echo -e "${YELLOW}[BUILD QUEUE] Cleaning up stale lock (PID $lock_pid died, age ${lock_age}s)${NC}"
            rm -f "$LOCK_FILE"
            continue
        fi
    fi
    
    # Show queue status periodically
    if [[ $((wait_duration % 10)) -eq 0 ]]; then
        show_queue_status
        
        # Show detailed wait metrics every 30s
        if [[ $((wait_duration % 30)) -eq 0 ]] && [[ $wait_duration -gt 0 ]]; then
            echo -e "${YELLOW}[METRICS] Queue wait time: ${wait_duration}s / ${MAX_WAIT_TIME}s max${NC}"
            if [[ -f "$LOCK_FILE" ]]; then
                lock_info=$(cat "$LOCK_FILE")
                lock_time=$(echo "$lock_info" | cut -d'|' -f3)
                build_duration=$((current_time - lock_time))
                echo -e "${YELLOW}[METRICS] Current build duration: ${build_duration}s / ${BUILD_TIMEOUT}s max${NC}"
            fi
        fi
    fi
    
    sleep $POLL_INTERVAL
done

# Remove ourselves from queue
rm -f "$QUEUE_DIR/$QUEUE_ID"

# Execute the build command with timeout
echo -e "${GREEN}[BUILD QUEUE] Executing: $BUILD_COMMAND${NC}"
total_wait_time=$(($(date +%s) - wait_start))
echo -e "${BLUE}[METRICS] Total queue wait time: ${total_wait_time}s${NC}"
build_start=$(date +%s)

# Execute with timeout
if timeout $BUILD_TIMEOUT bash -c "$BUILD_COMMAND"; then
    build_end=$(date +%s)
    build_duration=$((build_end - build_start))
    total_time=$((build_end - wait_start))
    echo -e "${GREEN}[BUILD QUEUE] Build completed successfully in ${build_duration}s${NC}"
    echo -e "${BLUE}[METRICS] Total time (queue + build): ${total_time}s${NC}"
    exit_code=0
else
    exit_code=$?
    build_end=$(date +%s)
    build_duration=$((build_end - build_start))
    total_time=$((build_end - wait_start))
    
    if [[ $exit_code -eq 124 ]]; then
        echo -e "${RED}[BUILD QUEUE] Build timed out after ${BUILD_TIMEOUT}s${NC}"
        echo -e "${BLUE}[METRICS] Total time (queue + timeout): ${total_time}s${NC}"
    else
        echo -e "${RED}[BUILD QUEUE] Build failed with exit code $exit_code after ${build_duration}s${NC}"
        echo -e "${BLUE}[METRICS] Total time (queue + build): ${total_time}s${NC}"
    fi
fi

# Lock will be released by cleanup function
exit $exit_code