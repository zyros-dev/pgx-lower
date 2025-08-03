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
        /home/xzel/repos/pgx-lower/tools/build_queue.sh bash -c "make utest && ./build/mlir_unit_test $*"
        ;;
    ptest)
        echo "Building for PostgreSQL tests (NOT queued - regression tests handle their own coordination)..."
        make ptest "$@"
        ;;
    build)
        echo "Building everything..."
        /home/xzel/repos/pgx-lower/tools/build_queue.sh make all "$@"
        ;;
    clean)
        echo "Cleaning build artifacts..."
        /home/xzel/repos/pgx-lower/tools/build_queue.sh make clean "$@"
        ;;
    compile_commands)
        echo "Generating compile_commands.json..."
        /home/xzel/repos/pgx-lower/tools/qcompile_commands.sh
        ;;
    dialects)
        echo "Building dialects..."
        /home/xzel/repos/pgx-lower/tools/build_queue.sh make -C build src/dialects "$@"
        ;;
    extension)
        echo "Building PostgreSQL extension..."
        /home/xzel/repos/pgx-lower/tools/build_queue.sh make -C extension "$@"
        ;;
    *)
        echo "Unknown operation: $operation"
        show_usage
        exit 1
        ;;
esac