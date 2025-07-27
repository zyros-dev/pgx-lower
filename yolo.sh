#!/bin/bash
# YOLO mode launcher for Claude

# Example usage:
# ./yolo.sh "implement the next lowering pass"
# ./yolo.sh --continue "fix the compilation errors"

# Set long timeouts for autonomous work sessions (20 hours = 72000000ms)
export BASH_DEFAULT_TIMEOUT_MS=72000000
export BASH_MAX_TIMEOUT_MS=72000000
export MCP_TIMEOUT=72000000

if [[ "$1" == "--continue" ]]; then
    shift
    claude --continue --print "$*" --dangerously-skip-permissions --verbose --output-format stream-json | jq
else
    claude --print "$*" --dangerously-skip-permissions --verbose --output-format stream-json | jq
fi