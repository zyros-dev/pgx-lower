#!/bin/bash
set -e

# Configuration
BUILD_DIR="build"

# Build the project with CMake
cmake -S . -B $BUILD_DIR -G Ninja
cmake --build $BUILD_DIR

# Run all regression tests with CTest
cd $BUILD_DIR
ctest --output-on-failure
cd -

echo "Done!"
