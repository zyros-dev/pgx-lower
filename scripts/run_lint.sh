#!/usr/bin/env bash
set -euo pipefail

# clang-tidy gate for the pgx-lower surface.
#
# Runs inside the builder container (see the `lint`/`lint-fix`/`lint-diff`
# recipes). Vendored code under src/lingodb/ is exempt via its own
# src/lingodb/.clang-tidy (Checks: '-*'); everything under src/pgx-lower/
# inherits the repo-root .clang-tidy.
#
# Usage: run_lint.sh [gate|inventory|fix|diff]
#   gate       any diagnostic fails the build (-warnings-as-errors='*')   [default]
#   inventory  no failure; prints total + per-check histogram (Task 4)
#   fix        auto-applies the mechanical fixes (clang-tidy --fix)
#   diff       gate, restricted to .cpp changed vs origin/main

MODE="${1:-gate}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# The compile DB. Reuse an existing build's compile_commands.json (exported
# by the configure/build recipes via -DCMAKE_EXPORT_COMPILE_COMMANDS=ON);
# otherwise stand up a lint-only build dir. Generated headers (tablegen, etc.)
# must exist for clang-tidy to parse, so a fresh lint dir is configured AND
# built once. build-artifacts/ is a named volume, so this persists on thor.
LINT_BUILD="$REPO_ROOT/build-artifacts/lint"

find_compile_db() {
    for d in release lint debug relwithdebinfo; do
        if [[ -f "$REPO_ROOT/build-artifacts/$d/compile_commands.json" ]]; then
            echo "$REPO_ROOT/build-artifacts/$d"
            return 0
        fi
    done
    return 1
}

ensure_compile_db() {
    if BUILD_DIR="$(find_compile_db)"; then
        echo "[run_lint] using compile DB: $BUILD_DIR/compile_commands.json" >&2
        return 0
    fi
    echo "[run_lint] no compile DB found; configuring + building $LINT_BUILD" >&2
    cmake -S "$REPO_ROOT" -B "$LINT_BUILD" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache
    # Build so generated headers exist for clang-tidy to parse.
    cmake --build "$LINT_BUILD"
    BUILD_DIR="$LINT_BUILD"
    if [[ ! -f "$BUILD_DIR/compile_commands.json" ]]; then
        echo "[run_lint] ERROR: compile_commands.json not produced in $BUILD_DIR" >&2
        exit 2
    fi
    echo "[run_lint] using compile DB: $BUILD_DIR/compile_commands.json" >&2
}

# Files the gate covers: pgx-lower .cpp only. lingodb is carved out by its
# own .clang-tidy, but we also never hand its files to the runner.
gate_files() {
    find "$REPO_ROOT/src/pgx-lower" -type f -name '*.cpp' \
        ! -path '*/build/*' | sort
}

diff_files() {
    git config --global --add safe.directory "$REPO_ROOT" >/dev/null 2>&1 || true
    git -C "$REPO_ROOT" diff --name-only origin/main...HEAD -- 'src/pgx-lower/**/*.cpp' 2>/dev/null \
        | sed "s#^#$REPO_ROOT/#" \
        | while read -r f; do [[ -f "$f" ]] && echo "$f"; done
}

ensure_compile_db

case "$MODE" in
gate)
    mapfile -t FILES < <(gate_files)
    echo "[run_lint] gate: ${#FILES[@]} files, -warnings-as-errors='*'" >&2
    run-clang-tidy -p "$BUILD_DIR" -warnings-as-errors='*' -quiet "${FILES[@]}"
    ;;
diff)
    mapfile -t FILES < <(diff_files)
    if [[ "${#FILES[@]}" -eq 0 ]]; then
        echo "[run_lint] diff: no pgx-lower .cpp changed vs origin/main; nothing to lint" >&2
        exit 0
    fi
    echo "[run_lint] diff: ${#FILES[@]} changed files, -warnings-as-errors='*'" >&2
    run-clang-tidy -p "$BUILD_DIR" -warnings-as-errors='*' -quiet "${FILES[@]}"
    ;;
fix)
    mapfile -t FILES < <(gate_files)
    echo "[run_lint] fix: applying clang-tidy --fix over ${#FILES[@]} files" >&2
    run-clang-tidy -p "$BUILD_DIR" -fix -quiet "${FILES[@]}"
    ;;
inventory)
    mapfile -t FILES < <(gate_files)
    echo "[run_lint] inventory: scanning ${#FILES[@]} files (advisory, non-failing)" >&2
    raw="$(run-clang-tidy -p "$BUILD_DIR" -quiet "${FILES[@]}" 2>/dev/null || true)"
    # Diagnostics look like:  path:line:col: warning: msg [check-name]
    # A line can carry multiple [a,b] aliases; count each.
    hits="$(printf '%s\n' "$raw" \
        | grep -oE '\[[a-z0-9]+[a-z0-9.-]*(,[a-z0-9.-]+)*\]$' \
        | tr -d '[]' | tr ',' '\n' | grep -E '.' || true)"
    total="$(printf '%s\n' "$hits" | grep -cE '.' || true)"
    echo "=== pgx-lower clang-tidy inventory ==="
    echo "total diagnostics: $total"
    echo "--- per-check histogram (desc) ---"
    printf '%s\n' "$hits" | sort | uniq -c | sort -rn
    ;;
*)
    echo "[run_lint] unknown mode: $MODE (use gate|inventory|fix|diff)" >&2
    exit 2
    ;;
esac
