#!/bin/bash
#
# Queued Build - Smart wrapper for pgx-lower MLIR pipeline build operations
# Architecture: PostgreSQL AST → RelAlg → DB → DSA → Standard MLIR → LLVM IR → JIT
#

export CLAUDE_AGENT_NAME="${CLAUDE_AGENT_NAME:-$(basename "$0")}"

show_usage() {
    echo "Usage: $0 <operation> [args...]"
    echo ""
    echo "Operations:"
    echo "  utest           - Build and run unit tests (MLIR dialects, lowering passes, streaming translators)"
    echo "  ptest           - Build PostgreSQL extension for regression tests (Test 1: SELECT * FROM test)"
    echo "  build           - Build complete MLIR pipeline (RelAlg, DB, DSA dialects + conversions)"
    echo "  clean           - Clean all build artifacts and CMake cache"
    echo "  compile_commands - Generate compile_commands.json for IDE support"
    echo "  dialects        - Build MLIR dialects only (RelAlg, DB, DSA, Util)"
    echo "  extension       - Build PostgreSQL extension with MLIR JIT compilation"
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
        echo "Building and running unit tests (MLIR dialects + lowering passes)..."
        # Build and test complete pipeline: RelAlg → DB → DSA → Standard MLIR
        /home/xzel/repos/pgx-lower/tools/build_queue.sh "make utest"
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "ERROR: Unit test build/run failed with exit code $exit_code"
            exit $exit_code
        fi
        ;;
    ptest)
        echo "Building PostgreSQL extension for regression tests (NOT queued)..."
        echo "Target: Test 1 'SELECT * FROM test' through complete MLIR → LLVM → JIT pipeline"
        make ptest "$@"
        ;;
    build)
        echo "Building complete MLIR pipeline (all dialects + conversion passes)..."
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
        echo "Building MLIR dialects (RelAlg, DB, DSA, Util)..."
        /home/xzel/repos/pgx-lower/tools/build_queue.sh make -C build src/dialects "$@"
        exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            echo "ERROR: Dialects build failed with exit code $exit_code"
            exit $exit_code
        fi
        ;;
    extension)
        echo "Building PostgreSQL extension with MLIR JIT compilation..."
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