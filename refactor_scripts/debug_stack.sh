#!/bin/bash

cd build-utest

# Start the test in background
./mlir_unit_test --gtest_filter="CircularTypeFixTest.UnconvertedNullableType" &
TEST_PID=$!

# Wait for it to start hanging (after debug 10.6)
sleep 5

echo "Attaching GDB to process $TEST_PID..."

# Attach GDB and get stack trace
gdb -batch \
    -ex "set confirm off" \
    -ex "attach $TEST_PID" \
    -ex "thread apply all bt 20" \
    -ex "info threads" \
    -ex "detach" \
    -ex "quit" \
    --pid=$TEST_PID

# Kill the hanging process
kill -9 $TEST_PID 2>/dev/null

echo "Stack trace capture complete"