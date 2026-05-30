#!/usr/bin/env bash
# clang-tidy gate for pgx-lower. Runs inside the pgx-lower-dev container
# (invoked from the lint/lint-fix/lint-diff/lint-inventory just recipes via
# `tsp docker exec`). Scopes to src/pgx-lower/*.cpp only; src/lingodb/ is
# carved out by its own .clang-tidy ('-*').
#
# Usage: run_lint.sh <WDIR> <MODE>
#   WDIR  workspace dir (the container path, e.g. /workspace)
#   MODE  check (default) | fix | diff | inventory
#     check      binary gate, -warnings-as-errors='*' — any diagnostic fails
#     fix        clang-tidy --fix (auto-fix pass, no gate)
#     diff       gate, only .cpp changed vs origin/main
#     inventory  advisory: total + per-check histogram, never fails
set -o pipefail
WDIR="${1:-/workspace}"
MODE="${2:-check}"
cd "$WDIR" || { echo "LINT: cannot cd $WDIR"; exit 2; }

# Dedicated lint build dir under build-artifacts/ (mutagen-ignored, sync-safe).
# Configure with compile-commands export, then build once: clang-tidy needs the
# generated MLIR TableGen .inc headers to exist. ccache makes re-runs cheap.
LINT_DIR="$WDIR/build-artifacts/lint"
mkdir -p "$LINT_DIR"
( cd "$LINT_DIR" \
  && { [ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug \
        -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "$WDIR"; } \
  && cmake --build . ) || { echo "LINT: configure/build failed"; exit 2; }

# Files to lint. `check`/`fix`/`inventory` cover the whole pgx-lower surface;
# `diff` narrows to .cpp changed vs origin/main.
if [ "$MODE" = "diff" ]; then
    git config --global --add safe.directory "$WDIR" >/dev/null 2>&1 || true
    git fetch origin main --quiet 2>/dev/null || true
    base=$(git merge-base origin/main HEAD 2>/dev/null || echo "")
    if [ -z "$base" ]; then echo "LINT: cannot resolve origin/main merge-base"; exit 2; fi
    mapfile -t files < <(git diff --name-only "$base" -- 'src/pgx-lower/*.cpp' 2>/dev/null | while read -r f; do [ -f "$f" ] && echo "$f"; done)
    if [ "${#files[@]}" -eq 0 ]; then
        echo "LINT: no src/pgx-lower/*.cpp changed vs origin/main — nothing to lint."
        exit 0
    fi
else
    mapfile -t files < <(find src/pgx-lower -name '*.cpp' | sort)
    if [ "${#files[@]}" -eq 0 ]; then echo "LINT: no src/pgx-lower/*.cpp found"; exit 2; fi
fi

case "$MODE" in
check|diff)
    rc=0
    printf '%s\n' "${files[@]}" \
      | xargs -P"$(nproc)" -I{} clang-tidy-20 -p "$LINT_DIR" --quiet -warnings-as-errors='*' {} \
      || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "LINT CLEAN — ${#files[@]} file(s) in src/pgx-lower/ pass all enabled checks"
    else
        echo "LINT FAILED — violations above (exit $rc). Fix them, or delete the offending rule from .clang-tidy."
    fi
    exit "$rc"
    ;;
fix)
    rc=0
    # SERIAL (no -P): clang-tidy --fix rewrites source in place; running it in
    # parallel races on shared headers and textually corrupts files (observed:
    # mangled if/else, broken braced-init). Correctness over speed for --fix.
    printf '%s\n' "${files[@]}" \
      | xargs -I{} clang-tidy-20 -p "$LINT_DIR" --quiet --fix --fix-errors {} \
      || rc=$?
    echo "LINT FIX applied over ${#files[@]} file(s). Review with 'git diff', then run 'just test'."
    exit 0
    ;;
inventory)
    # Advisory only: never fail. Diagnostics end in [check-name] (possibly
    # comma-joined aliases); split and tally each.
    raw=$(printf '%s\n' "${files[@]}" \
      | xargs -P"$(nproc)" -I{} clang-tidy-20 -p "$LINT_DIR" --quiet {} 2>/dev/null || true)
    hits=$(printf '%s\n' "$raw" \
      | grep -oE '\[[a-z0-9]+[a-z0-9.-]*(,[a-z0-9.-]+)*\]$' \
      | tr -d '[]' | tr ',' '\n' | grep -E '.')
    total=$(printf '%s\n' "$hits" | grep -cE '.')
    echo "=== pgx-lower clang-tidy inventory (${#files[@]} files) ==="
    echo "total diagnostics: $total"
    echo "--- per-check histogram (desc) ---"
    printf '%s\n' "$hits" | sort | uniq -c | sort -rn
    exit 0
    ;;
*)
    echo "LINT: unknown MODE '$MODE' (use check|fix|diff|inventory)"; exit 2 ;;
esac
