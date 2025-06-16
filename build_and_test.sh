#!/bin/bash
set -e

# Configuration
BUILD_DIR="build"
sudo rm -rfd $BUILD_DIR

# Build the project with CMake
cmake -S . -B $BUILD_DIR -G Ninja
sudo cmake --build $BUILD_DIR

sudo cmake --install $BUILD_DIR

# Run all regression tests with CTest
cd $BUILD_DIR
ctest --output-on-failure
cd -

echo "Done!"
