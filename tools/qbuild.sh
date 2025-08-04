#!/bin/bash
#
# Queued Build - Smart wrapper for common build operations
#

export CLAUDE_AGENT_NAME="${CLAUDE_AGENT_NAME:-$(basename "$0")}"

show_usage() {
    echo "Usage: $0 <operation> [args...]"
    echo ""
    echo "Operations:"
    echo "  utest           - Build and run unit tests"
    echo "  ptest           - Build for PostgreSQL regression tests (no queue)"
    echo "  build           - Build everything (make all)"
    echo "  clean           - Clean build artifacts"
    echo "  compile_commands - Generate compile_commands.json"
    echo "  dialects        - Build just the dialects"
    echo "  extension       - Build the PostgreSQL extension"
    echo ""
    echo "Examples:"
    echo "  $0 utest"
    echo "  $0 build -j4"
    echo "  $0 dialects"
}

if [[ $# -eq 0 ]]; then
    show_usage
    exit 1
fi

operation="$1"
shift

case "$operation" in
    utest)
        echo "Building and running unit tests..."
        # Use the correct Makefile target for unit tests
        /home/xzel/repos/pgx-lower/tools/build_queue.sh "make utest"
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "ERROR: Unit test build/run failed with exit code $exit_code"
            exit $exit_code
        fi
        ;;
    ptest)
        echo "Building for PostgreSQL tests (NOT queued - regression tests handle their own coordination)..."
        make ptest "$@"
        ;;
    build)
        echo "Building everything..."
        /home/xzel/repos/pgx-lower/tools/build_queue.sh make all "$@"
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "ERROR: Build failed with exit code $exit_code"
            exit $exit_code
        fi
        ;;
    clean)
        echo "Cleaning build artifacts..."
        /home/xzel/repos/pgx-lower/tools/build_queue.sh make clean "$@"
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "ERROR: Clean failed with exit code $exit_code"
            exit $exit_code
        fi
        ;;
    compile_commands)
        echo "Generating compile_commands.json..."
        /home/xzel/repos/pgx-lower/tools/qcompile_commands.sh
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "ERROR: compile_commands.json generation failed with exit code $exit_code"
            exit $exit_code
        fi
        ;;
    dialects)
        echo "Building dialects..."
        /home/xzel/repos/pgx-lower/tools/build_queue.sh make -C build src/dialects "$@"
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "ERROR: Dialects build failed with exit code $exit_code"
            exit $exit_code
        fi
        ;;
    extension)
        echo "Building PostgreSQL extension..."
        /home/xzel/repos/pgx-lower/tools/build_queue.sh make -C extension "$@"
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "ERROR: Extension build failed with exit code $exit_code"
            exit $exit_code
        fi
        ;;
    *)
        echo "Unknown operation: $operation"
        show_usage
        exit 1
        ;;
esac