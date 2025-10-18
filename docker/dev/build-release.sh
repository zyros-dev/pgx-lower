#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "Building release extension in dev container..."

# Check if container is running
if ! docker compose -f "$DOCKER_DIR/docker-compose.yml" ps | grep -q "pgx-lower-dev.*Up"; then
    echo -e "Error: pgx-lower-dev container is not running"
    echo "Start it with: cd docker && docker compose up -d dev"
    exit 1
fi

echo -e "Building with CMAKE_BUILD_TYPE=RelWithDebInfo (Release has optimizer bugs)"

# Execute RelWithDebInfo build inside the container
docker compose -f "$DOCKER_DIR/docker-compose.yml" exec dev bash -c "
    cd /workspace && \
    rm -rf build-docker-release && \
    mkdir -p build-docker-release && \
    cd build-docker-release && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_ONLY_EXTENSION=ON .. && \
    ninja && \
    strip --strip-debug extension/pgx_lower.so
"

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo -e "Release build completed: build-docker-release/extension/pgx_lower.so"
else
    echo -e "Release build failed with exit code $exit_code"
fi

exit $exit_code
