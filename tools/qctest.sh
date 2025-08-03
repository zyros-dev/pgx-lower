#!/bin/bash
#
# Queued CTest - Wrapper for ctest (unit tests) through build queue
#

export CLAUDE_AGENT_NAME="${CLAUDE_AGENT_NAME:-$(basename "$0")}"
exec /home/xzel/repos/pgx-lower/tools/build_queue.sh ctest "$@"