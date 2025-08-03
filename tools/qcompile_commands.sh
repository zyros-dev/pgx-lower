#!/bin/bash
#
# Queued Compile Commands - Generate compile_commands.json through build queue
#

export CLAUDE_AGENT_NAME="${CLAUDE_AGENT_NAME:-$(basename "$0")}"

# Change to build directory and generate compile commands
cd /home/xzel/repos/pgx-lower
exec /home/xzel/repos/pgx-lower/tools/build_queue.sh bash -c "cd build && cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cp compile_commands.json .."