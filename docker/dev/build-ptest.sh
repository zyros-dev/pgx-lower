#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="build-docker-ptest"

if ! docker ps --format '{{.Names}}' | grep -q "^pgx-lower-dev$"; then
    echo "Error: pgx-lower-dev container not running"
    echo "Start with: cd docker && docker compose up -d dev"
    exit 1
fi

echo "Building ptest in Docker container (${BUILD_DIR}/)"

# Build and test in container
docker exec pgx-lower-dev bash -c "
    cd /workspace && \
    rm -f CMakeCache.txt && \
    mkdir -p ${BUILD_DIR} && \
    cd ${BUILD_DIR} && \
    rm -f CMakeCache.txt && \
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Debug \
        -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
        -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
        -DPostgreSQL_CONFIG_EXECUTABLE=/usr/local/pgsql/bin/pg_config \
        .. && \
    ninja && \
    cd extension && \
    make clean && \
    make && \
    make install && \
    make installcheck
"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "Build completed successfully"
else
    echo "Build failed (exit code $exit_code)"
    echo "Check: ${BUILD_DIR}/extension/regression.diffs"
fi

exit $exit_code
