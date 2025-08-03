#!/bin/bash
#
# Queued Make - Wrapper for make commands through build queue
#

export CLAUDE_AGENT_NAME="${CLAUDE_AGENT_NAME:-$(basename "$0")}"
exec /home/xzel/repos/pgx-lower/tools/build_queue.sh make "$@"