#!/bin/bash
# Test compilation to find missing dependencies

echo "Testing compilation of pgx-lower..."
echo "Current directory: $(pwd)"
echo ""

echo "Step 1: Running CMake configuration..."
if cmake -S . -B build-test -G Ninja -DCMAKE_BUILD_TYPE=Debug 2>&1; then
    echo "✓ CMake configuration succeeded"
else
    echo "✗ CMake configuration failed - missing dependencies"
    exit 1
fi

echo ""
echo "Step 2: Building project..."
if cmake --build build-test 2>&1 | head -50; then
    echo "✓ Build started successfully"
else
    echo "✗ Build failed"
    exit 1
fi