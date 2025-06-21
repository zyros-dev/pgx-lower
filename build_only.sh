#!/bin/bash
set -e

BUILD_DIR="build"
rm -rf $BUILD_DIR

cmake -S . -B $BUILD_DIR -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build $BUILD_DIR

echo "Build completed!"