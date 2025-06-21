#!/bin/bash
set -e

# Configuration
BUILD_DIR="build"
sudo rm -rfd $BUILD_DIR

# Build the project with CMake
cmake -S . -B $BUILD_DIR -G Ninja -DCMAKE_BUILD_TYPE=Debug
sudo cmake --build $BUILD_DIR

sudo cmake --install $BUILD_DIR

echo "Done!"
