#!/bin/bash

# Monitor unit test memory usage
TEST_EXECUTABLE="/home/xzel/repos/pgx-lower/build-utest/mlir_unit_test"
LOG_FILE="/tmp/unit_test_memory.log"

echo "Starting unit test memory monitoring..." | tee "$LOG_FILE"
echo "Timestamp,PID,RSS(MB),VSZ(MB),CPU%" | tee -a "$LOG_FILE"

# Start unit tests in background
timeout 300s "$TEST_EXECUTABLE" &
TEST_PID=$!

echo "Unit test PID: $TEST_PID" | tee -a "$LOG_FILE"

# Monitor every second
while kill -0 $TEST_PID 2>/dev/null; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    MEMORY_INFO=$(ps -p $TEST_PID -o rss,vsz,%cpu --no-headers 2>/dev/null)
    
    if [ -n "$MEMORY_INFO" ]; then
        RSS_KB=$(echo "$MEMORY_INFO" | awk '{print $1}')
        VSZ_KB=$(echo "$MEMORY_INFO" | awk '{print $2}')  
        CPU=$(echo "$MEMORY_INFO" | awk '{print $3}')
        
        RSS_MB=$((RSS_KB / 1024))
        VSZ_MB=$((VSZ_KB / 1024))
        
        echo "$TIMESTAMP,$TEST_PID,${RSS_MB},${VSZ_MB},${CPU}%" | tee -a "$LOG_FILE"
        
        # Check for excessive memory usage
        if [ $RSS_MB -gt 10000 ]; then
            echo "WARNING: Memory usage exceeded 10GB, killing process" | tee -a "$LOG_FILE"
            kill -9 $TEST_PID
            break
        fi
    fi
    
    sleep 1
done

wait $TEST_PID 2>/dev/null
EXIT_CODE=$?

echo "Unit tests finished with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "Memory monitoring log saved to: $LOG_FILE"