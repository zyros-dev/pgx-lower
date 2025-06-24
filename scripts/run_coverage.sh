#!/bin/bash

# Script to run unit tests with coverage and generate report

set -e

echo "Building with coverage enabled..."

# Clean previous coverage data
rm -f *.profraw *.profdata
rm -rf coverage_report/

# Build with coverage
cmake -S . -B build-coverage -G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
cmake --build build-coverage

echo "Running unit tests..."

# Set profile file name
export LLVM_PROFILE_FILE="coverage.profraw"

# Run unit tests
cd build-coverage && ./mlir_unit_test && cd -

echo "Generating coverage report..."

# Check if we have coverage data
if [ ! -f coverage.profraw ]; then
    echo "No coverage data generated!"
    exit 1
fi

# Convert raw profile data to indexed format
llvm-profdata merge -sparse coverage.profraw -o coverage.profdata

# Generate HTML coverage report
mkdir -p coverage_report
llvm-cov show build-coverage/mlir_unit_test -instr-profile=coverage.profdata \
    -format=html -output-dir=coverage_report \
    -ignore-filename-regex="build-.*|third_party|.*gtest.*"

# Generate summary
llvm-cov report build-coverage/mlir_unit_test -instr-profile=coverage.profdata \
    -ignore-filename-regex="build-.*|third_party|.*gtest.*"

echo "Coverage report generated in coverage_report/"
echo "Open coverage_report/index.html in a browser to view the report"