#!/bin/bash
#
# Queued Unit Test - Wrapper for common unit test commands through build queue
#

export CLAUDE_AGENT_NAME="${CLAUDE_AGENT_NAME:-$(basename "$0")}"

# Default to running all unit tests
if [[ $# -eq 0 ]]; then
    exec /home/xzel/repos/pgx-lower/tools/build_queue.sh ./build/mlir_unit_test
else
    exec /home/xzel/repos/pgx-lower/tools/build_queue.sh ./build/mlir_unit_test "$@"
fi