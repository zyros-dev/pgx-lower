#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "$Starting containerized ptest build..."

if ! docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps | grep -q "pgx-lower-dev.*Up"; then
    echo -e "Error: pgx-lower-dev container is not running"
    echo "Start it with: cd docker && docker compose up -d dev"
    exit 1
fi

echo -e "Executing: ./tools/qbuild.sh ptest"

docker compose -f "$SCRIPT_DIR/docker-compose.yml" exec dev bash -c "cd /workspace && ./tools/qbuild.sh ptest"

exit_code=$?

if [[ $exit_code -eq 0 ]]; then
    echo -e "Container ptest build completed successfully"
else
    echo -e "Container ptest build failed with exit code $exit_code"
fi

exit $exit_code
