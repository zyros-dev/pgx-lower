#!/bin/bash
#
# Queued CMake - Wrapper for cmake build commands through build queue
#

export CLAUDE_AGENT_NAME="${CLAUDE_AGENT_NAME:-$(basename "$0")}"
exec /home/xzel/repos/pgx-lower/tools/build_queue.sh cmake "$@"