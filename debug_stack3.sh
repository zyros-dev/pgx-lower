#!/bin/bash

cd /home/xzel/repos/pgx-lower/build-utest

echo "Starting test..."
timeout 20 strace -e trace=none -c ./mlir_unit_test --gtest_filter="CircularTypeFixTest.UnconvertedNullableType" &
TEST_PID=$!

# Wait for it to hang 
sleep 10

echo "Getting process info for PID: $TEST_PID"
ps aux | grep $TEST_PID
echo ""

# Attach with full detailed stack trace
echo "Getting detailed stack trace..."
gdb -p $TEST_PID -batch \
    -ex "set pagination off" \
    -ex "bt 40" \
    -ex "info registers" \
    -ex "detach"

# Clean up
kill -9 $TEST_PID 2>/dev/null
wait $TEST_PID 2>/dev/null

echo "Debug complete"