#!/bin/bash

cd /home/xzel/repos/pgx-lower/build-utest

# Start the test directly (no timeout wrapper)
echo "Starting test..."
./mlir_unit_test --gtest_filter="CircularTypeFixTest.UnconvertedNullableType" &
TEST_PID=$!

echo "Test started with PID: $TEST_PID"

# Wait for it to hang (after debug 10.6)
sleep 10

# Check if process is still running
if ! kill -0 $TEST_PID 2>/dev/null; then
    echo "Process already finished or crashed"
    exit 1
fi

echo "Process is still running - likely hanging in infinite loop"
echo "Getting detailed stack trace..."

# Get very detailed stack trace
gdb -p $TEST_PID -batch \
    -ex "set pagination off" \
    -ex "set print pretty on" \
    -ex "set print frame-arguments all" \
    -ex "bt 50" \
    -ex "info threads" \
    -ex "thread apply all bt 10" \
    -ex "info stack" \
    -ex "detach"

# Clean up
echo "Killing hanging process..."
kill -9 $TEST_PID 2>/dev/null

echo "Debug complete"