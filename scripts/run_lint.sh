#!/usr/bin/env bash
# Lint gate for pgx-lower. Runs clang-tidy-20 over src/pgx-lower/*.cpp only;
# src/lingodb/ is carved out by its own .clang-tidy ('-*').
#
# Usage, inside the dev container:
#   bash scripts/run_lint.sh <workspace> <check|fix|inventory> [file...]
set -o pipefail

WDIR="${1:-/workspace}"
MODE="${2:-check}"
shift 2 || true

cd "$WDIR" || {
    echo "LINT: cannot cd $WDIR"
    exit 2
}

LINT_DIR="$WDIR/build-docker-lint"
mkdir -p "$LINT_DIR"

if [ "${LINT_SKIP_BUILD:-0}" = "1" ]; then
    if [ ! -f "$LINT_DIR/compile_commands.json" ]; then
        echo "LINT: $LINT_DIR/compile_commands.json missing; run just lint once first"
        exit 2
    fi
else
    # The build is load-bearing: clang-tidy needs the generated MLIR TableGen
    # headers, and ccache makes repeated configure/build passes cheap.
    (
        cd "$LINT_DIR" \
            && { [ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug \
                -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache "$WDIR"; } \
            && cmake --build .
    ) || {
        echo "LINT: configure/build failed"
        exit 2
    }
fi

if [ "$#" -gt 0 ]; then
    files=("$@")
    scope="selected file(s)"
else
    mapfile -t files < <(find src/pgx-lower -name '*.cpp' | sort)
    scope="file(s) in src/pgx-lower/"
fi
if [ "${#files[@]}" -eq 0 ]; then
    echo "LINT: no files selected"
    exit 2
fi

case "$MODE" in
check)
    tidy_args=(-p "$LINT_DIR" --quiet --allow-no-checks -warnings-as-errors='*')
    jobs="$(nproc)"
    ;;
fix)
    tidy_args=(-p "$LINT_DIR" --quiet --allow-no-checks --fix --fix-errors)
    jobs=1
    ;;
inventory)
    tidy_args=(-p "$LINT_DIR" --quiet --allow-no-checks)
    jobs="$(nproc)"
    ;;
*)
    echo "LINT: unknown MODE '$MODE'"
    exit 2
    ;;
esac

out="$(mktemp)"
rc=0
printf '%s\n' "${files[@]}" \
    | xargs -P"$jobs" -I{} clang-tidy-20 "${tidy_args[@]}" {} \
        >"$out" 2>&1 || rc=$?

sed '/^[0-9][0-9]* warnings generated\.$/d' "$out"

if [ "$MODE" = "inventory" ]; then
    echo ""
    echo "LINT INVENTORY — ${#files[@]} file(s)"
    total=$(grep -cE 'warning:|error:' "$out" || true)
    echo "total diagnostics: $total"
    grep -oE '\[[a-z0-9_.-]+(,[a-z0-9_.-]+)*\]$' "$out" \
        | sort \
        | uniq -c \
        | sort -rn || true
elif [ "$MODE" = "check" ]; then
    if [ "$rc" -eq 0 ]; then
        echo "LINT CLEAN — ${#files[@]} ${scope} pass all enabled checks"
    else
        echo "LINT FAILED — violations above (exit $rc). Fix them, or delete the offending rule from .clang-tidy."
    fi
fi

rm -f "$out"
exit "$rc"
