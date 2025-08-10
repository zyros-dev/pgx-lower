#!/bin/bash

cd /home/xzel/repos/pgx-lower/build-utest

# Start the test 
timeout 15 ./mlir_unit_test --gtest_filter="CircularTypeFixTest.UnconvertedNullableType" &

# Wait for it to hang
sleep 12

# Find the actual test process
TEST_PID=$(ps aux | grep "mlir_unit_test --gtest_filter" | grep -v grep | grep -v timeout | awk '{print $2}' | head -1)

echo "Found test process: $TEST_PID"

if [ ! -z "$TEST_PID" ]; then
    echo "Getting stack trace..."
    gdb -p $TEST_PID -batch -ex "set pagination off" -ex "bt 30" -ex "detach"
    
    # Kill the process
    kill -9 $TEST_PID
fi

echo "Done"