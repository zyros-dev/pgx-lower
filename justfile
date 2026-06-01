# pgx-lower developer commands.
# Every recipe that does real work runs on thor; the mac is an edit host.
# Builds/tests/bench run inside the pgx-lower-dev Docker container on thor,
# serialized through task-spooler (tsp) so concurrent agents queue instead of
# fighting for RAM. Run `just --list` to see everything.

_thor := "comfy"
_ctr  := "pgx-lower-dev"

# Working directory inside the container. Main repo maps to /workspace; a
# worktree at .worktrees/<name> on the host maps to /workspace/.worktrees/<name>
# (the container mounts the whole repo, so worktrees are visible automatically).
# Derived from the main repo root (parent of --git-common-dir) so it works
# whether just is invoked from the main checkout or a worktree with its own
# justfile copy.
_main_root := shell('dirname "$(git rev-parse --path-format=absolute --git-common-dir)"')
_rel  := replace_regex(invocation_directory(), "^" + _main_root + "/?", "")
_wdir := if _rel == "" { "/workspace" } else { "/workspace/" + _rel }
_bdir := _wdir + "/build-artifacts/ptest"
_bench_output := _wdir + "/build-artifacts/bench-output"
_thor_root := "/home/zel/repos/pgx-lower"
_thor_wdir := if _rel == "" { _thor_root } else { _thor_root + "/" + _rel }

# Serialized build queue: compile/test/bench share one slot on thor so they
# don't skew each other's timings or OOM. Check runs on a separate queue.
_build_q := "pgx-build"
_check_q := "pgx-check"

default:
    @just --list

# --- Preflight -------------------------------------------------------------

# Verify tsp + dev container are up on thor, and force mutagen to finish
# the mac→thor sync so recipes reading files on thor see the edits you
# just made locally. Flush is cheap (~100ms when nothing's pending) and
# removes the "sleep 3 before just compile" class of cargo-cult timing.
_preflight:
    @ssh {{_thor}} 'command -v tsp >/dev/null 2>&1 || { echo "ERROR: task-spooler not installed on thor. Run: ssh comfy sudo apt-get install -y task-spooler"; exit 1; }'
    @ssh {{_thor}} 'docker ps --format "{{{{.Names}}" | grep -q "^{{_ctr}}$" || { echo "ERROR: {{_ctr}} not running on thor. Run: just up"; exit 1; }'
    @mutagen sync flush "pgx-lower" >/dev/null 2>&1 || true

# Start the dev container on thor (one-time per boot).
up:
    ssh {{_thor}} 'cd ~/repos/pgx-lower/docker && docker compose up -d dev'

# Stop the dev container.
down:
    ssh {{_thor}} 'cd ~/repos/pgx-lower/docker && docker compose stop dev'

# Install task-spooler on thor (needs sudo; prompts on thor).
bootstrap-tsp:
    ssh -t {{_thor}} 'sudo apt-get install -y task-spooler'

# --- Build / check / test / bench -----------------------------------------

# Incremental build of the pgx_lower extension (queued). First run in a worktree
# configures cmake with ccache compiler launcher; later runs are incremental
# ninja + ccache, so new worktrees with no code changes link in <1min.
# Emits a one-line "BUILD OK" / "BUILD FAILED" verdict at the end so agents
# don't have to infer success by reading the full cmake/ninja scroll.
compile: _preflight
    #!/usr/bin/env bash
    set -o pipefail
    # Block until the mac→thor mutagen cycle completes so ninja sees the
    # file mtimes you just edited locally. Removes the "sleep 3 before
    # just compile" cargo-cult. We always work in the main checkout now,
    # so the session is always the bare "pgx-lower".
    mutagen sync flush "pgx-lower" >/dev/null 2>&1 || true
    ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "mkdir -p {{_bdir}} && cd {{_bdir}} && ([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache {{_wdir}}) && cmake --build . && cmake --install .") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id' 2>&1 | tee /tmp/pgx-compile.out
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        ninja_targets=$(grep -cE '^\[[0-9]+/[0-9]+\]' /tmp/pgx-compile.out 2>/dev/null || echo 0)
        echo ""
        echo "BUILD OK — ${ninja_targets} ninja step(s), pgx_lower.so installed"
        just _refresh-clion-db
    else
        errs=$(grep -cE 'error:|FAILED:' /tmp/pgx-compile.out 2>/dev/null || echo 0)
        echo ""
        echo "BUILD FAILED — ${errs} error line(s), exit $rc. Last 30 lines:"
        tail -n 30 /tmp/pgx-compile.out
        exit "$rc"
    fi

# Print ccache statistics from the dev container.
ccache-stats:
    @ssh {{_thor}} 'docker exec {{_ctr}} ccache --show-stats | head -20'

# Refresh the CLion-friendly compilation database on thor.
#
# CMake runs inside the Docker container, so the raw DB uses /workspace paths.
# CLion Gateway opens the repo on the thor host at /home/zel/repos/pgx-lower,
# so it needs those paths rewritten before it can attach files to targets.
# Refresh compile_commands.json for CLion Gateway.
_refresh-clion-db:
    #!/usr/bin/env bash
    set -euo pipefail
    ssh {{_thor}} 'set -euo pipefail
        sysroot="{{_thor_wdir}}/build-artifacts/clion-sysroot"
        if [ ! -f "${sysroot}/.llvm-20-headers-ready" ]; then
            rm -rf "${sysroot}/usr/lib/llvm-20/include"
            mkdir -p "${sysroot}/usr/lib/llvm-20"
            docker exec {{_ctr}} tar -chf - -C /usr/lib/llvm-20 include | tar -xf - -C "${sysroot}/usr/lib/llvm-20"
            touch "${sysroot}/.llvm-20-headers-ready"
        fi
        if [ ! -f "${sysroot}/.pgsql-headers-ready" ]; then
            rm -rf "${sysroot}/usr/local/pgsql/include"
            mkdir -p "${sysroot}/usr/local/pgsql"
            docker exec {{_ctr}} tar -chf - -C /usr/local/pgsql include | tar -xf - -C "${sysroot}/usr/local/pgsql"
            touch "${sysroot}/.pgsql-headers-ready"
        fi
        if [ ! -f "${sysroot}/.usr-include-ready" ]; then
            rm -rf "${sysroot}/usr/include"
            mkdir -p "${sysroot}/usr"
            docker exec {{_ctr}} tar -chf - -C /usr include | tar -xf - -C "${sysroot}/usr"
            touch "${sysroot}/.usr-include-ready"
        fi
        if [ ! -f "${sysroot}/.usr-local-include-ready" ]; then
            rm -rf "${sysroot}/usr/local/include"
            mkdir -p "${sysroot}/usr/local"
            docker exec {{_ctr}} tar -chf - -C /usr/local include | tar -xf - -C "${sysroot}/usr/local"
            touch "${sysroot}/.usr-local-include-ready"
        fi
        if [ ! -f "${sysroot}/.llvm-20-resource-headers-ready" ]; then
            rm -rf "${sysroot}/usr/lib/llvm-20/lib/clang/20/include"
            mkdir -p "${sysroot}/usr/lib/llvm-20/lib/clang/20"
            docker exec {{_ctr}} tar -chf - -C /usr/lib/llvm-20/lib/clang/20 include | tar -xf - -C "${sysroot}/usr/lib/llvm-20/lib/clang/20"
            touch "${sysroot}/.llvm-20-resource-headers-ready"
        fi'
    ssh {{_thor}} 'python3 - <<'"'"'PY'"'"'
    import json
    import shlex
    from pathlib import Path
    
    container_root = "/workspace"
    host_root = "{{_thor_wdir}}"
    sysroot = f"{host_root}/build-artifacts/clion-sysroot"
    src = Path("{{_thor_wdir}}/build-artifacts/ptest/compile_commands.json")
    dst = Path("{{_thor_wdir}}/compile_commands.json")
    wrappers = {
        "/usr/lib/llvm-20/bin/clang": [f"{host_root}/tools/clion/container-clang"],
        "/usr/lib/llvm-20/bin/clang++": [f"{host_root}/tools/clion/container-clang++"],
    }
    system_includes = [
        f"{sysroot}/usr/include/c++/14",
        f"{sysroot}/usr/include/x86_64-linux-gnu/c++/14",
        f"{sysroot}/usr/include/c++/14/backward",
        f"{sysroot}/usr/lib/llvm-20/lib/clang/20/include",
        f"{sysroot}/usr/local/include",
        f"{sysroot}/usr/include/x86_64-linux-gnu",
        f"{sysroot}/usr/include",
    ]
    
    with src.open() as f:
        entries = json.load(f)
    
    def rewrite(value):
        if not isinstance(value, str):
            return value
        value = value.replace(container_root, host_root)
        value = value.replace("/usr/lib/llvm-20/include", f"{sysroot}/usr/lib/llvm-20/include")
        value = value.replace("/usr/lib/llvm-20/lib/clang/20/include", f"{sysroot}/usr/lib/llvm-20/lib/clang/20/include")
        value = value.replace("/usr/local/include", f"{sysroot}/usr/local/include")
        value = value.replace("/usr/local/pgsql/include", f"{sysroot}/usr/local/pgsql/include")
        value = value.replace("/usr/include", f"{sysroot}/usr/include")
        return value

    def add_system_includes(argv):
        present = set()
        skip_next = False
        for i, arg in enumerate(argv):
            if skip_next:
                skip_next = False
                continue
            if arg in ("-I", "-isystem") and i + 1 < len(argv):
                present.add(argv[i + 1])
                skip_next = True
            elif arg.startswith("-I") and len(arg) > 2:
                present.add(arg[2:])
            elif arg.startswith("-isystem") and len(arg) > len("-isystem"):
                present.add(arg[len("-isystem"):])

        additions = []
        for path in system_includes:
            if path not in present:
                additions.extend(["-isystem", path])
        if len(argv) > 1 and Path(argv[0]).name.startswith("container-clang"):
            return [argv[0], *additions, *argv[1:]]
        return [*additions, *argv]
    
    def rewrite_argv(argv):
        rewritten = [rewrite(arg) for arg in argv]
        if rewritten and rewritten[0] in wrappers:
            rewritten = wrappers[rewritten[0]] + rewritten[1:]
        return add_system_includes(rewritten)
    
    for entry in entries:
        for key in ("directory", "file", "output"):
            if key in entry:
                entry[key] = rewrite(entry[key])
        if "command" in entry:
            entry["command"] = shlex.join(rewrite_argv(shlex.split(entry["command"])))
        if "arguments" in entry:
            entry["arguments"] = rewrite_argv(entry["arguments"])
    
    with dst.open("w") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")
    
    print(dst)
    PY'
    echo "CLion DB ready on thor: {{_thor_wdir}}/compile_commands.json"

# Force a build, then refresh the CLion database. Normal use should not need
# this; `just compile`, `just test`, and `just utest-pg` refresh it automatically.
clion-db: compile

# Fast static analysis: clang-format dry-run over the whole src/ tree.
# Parallel-safe — runs on the check queue, not the build queue.
# NOTE: the repo has hundreds of pre-existing violations. For gating PRs,
# prefer `just check-diff` which scopes to files you actually changed.
check: _preflight
    @ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_check_q}}.sock && id=$(tsp docker exec {{_ctr}} bash -c "cd {{_wdir}} && make fcheck") && echo "[job $id queued on {{_check_q}}]" && tsp -c $id'

# clang-format-diff on the exact hunks changed vs origin/main — the real PR
# gate. Unlike `just check`, which flags hundreds of pre-existing violations
# on any file you touch, this only surfaces formatting issues in lines your
# diff actually added/modified. Prints "check-diff: clean" and exits 0 when
# your hunks are properly formatted; prints the specific lines and exits 1
# otherwise. Safe to run from any branch.
check-diff: _preflight
    #!/usr/bin/env bash
    set -eo pipefail
    git fetch origin main --quiet
    base=$(git merge-base origin/main HEAD)
    diff=$(git diff -U0 "$base" -- 'src/*.c' 'src/*.cc' 'src/*.cpp' 'src/*.h' 'src/*.hpp' 'tests/*.c' 'tests/*.cc' 'tests/*.cpp' 'tests/*.h' 'tests/*.hpp' 'extension/*.c' 'extension/*.h' 2>/dev/null || true)
    if [ -z "$diff" ]; then
        echo "No C/C++ hunks changed vs origin/main — nothing to check."
        exit 0
    fi
    echo "Checking hunks changed vs origin/main..."
    # clang-format-diff reads a unified diff on stdin, applies style only
    # within the added/modified line ranges, and prints a replacement diff
    # for any lines that don't match the style. Empty output = clean.
    # `<<<` adds a trailing newline that clang-format-diff wants; pure
    # printf '%s' doesn't, and the upstream can SIGPIPE us before finishing.
    # Capture via file rather than $(...) so `set -e` doesn't abort on a
    # harmless non-zero exit from something downstream.
    ssh {{_thor}} "docker exec -i {{_ctr}} bash -c 'cd {{_wdir}} && clang-format-diff-20 -p1 -style=file'" <<<"$diff" > /tmp/check-diff.out || true
    if [ ! -s /tmp/check-diff.out ]; then
        echo "check-diff: clean (your hunks match the project style)"
        exit 0
    fi
    cat /tmp/check-diff.out
    echo ""
    echo "check-diff: your hunks need reformatting. Run \`just ffix-diff\` to auto-fix, or hand-edit the specific lines above."
    exit 1

# Apply clang-format-diff to the hunks this PR changed, in place. Fixes
# formatting inside your added/modified line ranges without touching
# pre-existing violations elsewhere in the file — safe to run on files with
# pre-existing debt because it scopes to your diff, not the whole file.
ffix-diff: _preflight
    #!/usr/bin/env bash
    set -euo pipefail
    git fetch origin main --quiet
    base=$(git merge-base origin/main HEAD)
    diff=$(git diff -U0 "$base" -- 'src/*.c' 'src/*.cc' 'src/*.cpp' 'src/*.h' 'src/*.hpp' 'tests/*.c' 'tests/*.cc' 'tests/*.cpp' 'tests/*.h' 'tests/*.hpp' 'extension/*.c' 'extension/*.h' 2>/dev/null || true)
    if [ -z "$diff" ]; then
        echo "No C/C++ hunks to fix."
        exit 0
    fi
    # Ensure files are synced (mutagen lag) then run fixer in-place on thor.
    ssh {{_thor}} "docker exec -i {{_ctr}} bash -c 'cd {{_wdir}} && clang-format-diff-20 -p1 -i -style=file'" <<<"$diff"
    echo "ffix-diff: formatted hunks in place. Review with 'git diff' and re-stage."

# Copy the authoritative pg_regress output for a test into tests/expected/,
# overwriting any hand-written version. This is the right way to build the
# .out file for a new regression test — don't write it by hand, because
# pg_regress's format (SQL echo lines, NOTICE messages, trailing whitespace
# on column headers) isn't what a plain psql session looks like, and most
# editors strip the trailing spaces on save anyway.
#
# Usage: just expected-from-results 43_version
# Requires: `just compile && just test` already ran for this branch — the
# recipe reads build-artifacts/ptest/extension/results/<name>.out on thor.
expected-from-results TEST:
    #!/usr/bin/env bash
    set -euo pipefail
    src="{{_bdir}}/extension/results/{{TEST}}.out"
    # Friendly precondition: when a first-time user (or a canary on a
    # fresh worktree) runs `just expected-from-results <name>` before
    # `just test`, the results file simply doesn't exist on thor yet.
    # Rather than error with an opaque "file not found" from the
    # subsequent cat, point them at the one step they need to take.
    # A RED `just test` run is enough — pg_regress still writes
    # results/<name>.out before bailing on the missing expected file.
    if ! ssh {{_thor}} "docker exec {{_ctr}} test -f ${src}" 2>/dev/null; then
        echo "expected-from-results: results/{{TEST}}.out not found on thor."
        echo "Run \`just test\` first to generate it (RED run is fine; the bail-out still produces the results/ output)."
        exit 1
    fi
    mkdir -p tests/expected
    ssh {{_thor}} "docker exec {{_ctr}} cat ${src}" > "tests/expected/{{TEST}}.out"
    echo "Wrote tests/expected/{{TEST}}.out ($(wc -l <tests/expected/{{TEST}}.out) lines, $(wc -c <tests/expected/{{TEST}}.out) bytes)."
    echo "Re-run 'just test' — it should now pass for this case."

# Run PostgreSQL regression tests (queued), gated against
# tests/pg_regress_baseline.txt. Exits non-zero only on *delta* vs the
# baseline (new failures, or previously-failing tests that now pass).
# ctest runs as the postgres user so pg_regress's default "whoami"
# connection works.
#
# ALWAYS rebuilds + reinstalls pgx_lower.so before running tests. The
# reinstall is load-bearing: `cmake --build` updates the .so inside the
# build dir, but pg_regress loads the COPY at /usr/local/pgsql/lib/
# pgx_lower.so. Without a `cmake --install`, a stale .so from a prior
# worktree or a prior spec can answer the test's queries with
# yesterday's symbols, silently GREENing a test that should be RED
# — the TDD-killer pattern. The build+install steps are no-ops when
# source hasn't changed (ccache + cmake timestamp compare), so doing
# them unconditionally is cheap and strictly safer. Configures cmake
# on first run in a fresh worktree, mirroring `just compile`.
#
# Note: this is the OUTPUT-EQUIVALENCE suite — it proves pgx_lower
# matches stock PG on a curated stable set of queries. For TDD on most
# spec work, prefer `just utest-pg` (faster, scoped to the thing you're
# actually changing). See SKILL.md step 2.
test: _preflight
    #!/usr/bin/env bash
    set -euo pipefail
    ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "mkdir -p {{_bdir}} && cd {{_bdir}} && ([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache {{_wdir}}) && cmake --build . && cmake --install . && mkdir -p /tmp/pgx_ir && chmod 777 /tmp/pgx_ir; chmod o+x /workspace/.worktrees 2>/dev/null || true; chmod -R o+rX {{_wdir}}; chown -R postgres:postgres {{_bdir}} && cd {{_bdir}} && (su postgres -c \"ctest -V\" 2>&1 | tee /tmp/ctest.out; cat /tmp/ctest.out | python3 {{_wdir}}/scripts/ptest_with_baseline.py --baseline-file {{_wdir}}/tests/pg_regress_baseline.txt)") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'
    just _refresh-clion-db

# Fast PG-aware unit tests (spec 16). Runs each .sql under tests/regress-unit/sql/
# with `psql -v ON_ERROR_STOP=on`. Each .sql is a DO block that PERFORMs the
# unit C functions linked into pgx_lower.so (Debug builds only). Failures
# trigger via elog(ERROR, ...) which makes psql exit non-zero.
utest-pg: _preflight
    #!/usr/bin/env bash
    set -euo pipefail
    python3 {{invocation_directory()}}/scripts/gen_unit_test_sql.py
    ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "export PATH=/usr/local/pgsql/bin:\$PATH && mkdir -p {{_bdir}} && cd {{_bdir}} && ([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache {{_wdir}}) && cmake --build . && cmake --install . && chmod o+x /workspace/.worktrees 2>/dev/null || true; chmod -R o+rX {{_wdir}}/tests/regress-unit && su postgres -c \"/usr/local/pgsql/bin/dropdb --if-exists regression_unit && /usr/local/pgsql/bin/createdb regression_unit\" && fail=0; for sql in {{_wdir}}/tests/regress-unit/sql/*.sql; do echo \"--- \$(basename \$sql) ---\"; su postgres -c \"/usr/local/pgsql/bin/psql -v ON_ERROR_STOP=on -d regression_unit -f \$sql\" || { fail=1; echo FAIL: \$sql; }; done; echo; if [ \$fail -eq 0 ]; then echo UTEST-PG_OK; else echo UTEST-PG_FAILED; exit 1; fi") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'
    just _refresh-clion-db

# Re-record the pg_regress baseline. Run only when you have consciously
# accepted a new set of red tests on main — each entry that gets added
# here must be justified in the PR body. Removing entries is free (those
# tests are now passing).
test-record-baseline: _preflight
    #!/usr/bin/env bash
    set -euo pipefail
    ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "mkdir -p /tmp/pgx_ir && chmod 777 /tmp/pgx_ir; chmod o+x /workspace/.worktrees 2>/dev/null || true; chmod -R o+rX {{_wdir}}; chown -R postgres:postgres {{_bdir}} && cd {{_bdir}} && (su postgres -c \"ctest -V\" 2>&1 | tee /tmp/ctest.out; cat /tmp/ctest.out | python3 {{_wdir}}/scripts/ptest_with_baseline.py --baseline-file {{_wdir}}/tests/pg_regress_baseline.txt --record)") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

# Smoke benchmark: SF=0.5, 1 iteration per query, pgx ON vs OFF.
# First run on a fresh container takes ~5 min (dbgen + psql load dominates);
# subsequent runs take ~2 min because run.py detects TPC-H is already loaded
# at SF=0.5 (customer row count == 75_000) and skips the regen+reload step.
# Running at SF=0.5 puts us in the execution-dominated regime (~88% of each
# query's wall time is real execution, ~12% is JIT compile), so the per-query
# variance drops from the ~50% bimodal mess we saw at SF=0.01 to honest
# 5–10% execution noise. iter=1 is enough at this SF — medianing multiple
# iterations doesn't meaningfully reduce execution variance, it just burns
# time.
#
# Recreates build-artifacts/bench-output/ each run — a partial interrupted earlier run
# can leave sqlite journal/lock state that makes subsequent connects open
# readonly and bomb mid-run with "attempt to write a readonly database".
bench: _preflight
    @ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "rm -rf {{_bench_output}} && mkdir -p {{_bench_output}} && chmod 777 {{_bench_output}} && cd {{_wdir}} && python3 benchmark/tpch/run.py 0.5 --port 5432 --container {{_ctr}} --indexes --skip q17,q20 --iterations 1") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

# Deeper-signal benchmark: SF=1, 1 iteration. ~10 min first time, ~6 min
# cached. Run before merging anything that claims a performance improvement
# where SF=0.5's numbers feel marginal. At SF=1 the per-query wall time is
# long enough (seconds) that ±5% is real signal.
bench-merge: _preflight
    @ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "rm -rf {{_bench_output}} && mkdir -p {{_bench_output}} && chmod 777 {{_bench_output}} && cd {{_wdir}} && python3 benchmark/tpch/run.py 1.0 --port 5432 --container {{_ctr}} --indexes --skip q17,q20 --iterations 1") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

# Generate the PR benchmark report. Requires an open PR (the PR number
# becomes part of the filename). Snapshots the current benchmark.db to
# ./bench-results/pr-<N>-<branch>.db, pulls the baseline db directly from
# origin/main (not committed to feature branches), and emits matching
# .png + .md.
#
# Baseline: the alphanumerically latest .db in origin/main:bench-results/
# (which is the most recently merged PR's db). Baseline dbs never land on
# feature branches — each PR commits exactly one .db (its own).
bench-report:
    #!/usr/bin/env bash
    set -euo pipefail
    branch=$(git rev-parse --abbrev-ref HEAD)
    pr=$(gh pr view --json number -q .number 2>/dev/null || true)
    if [ -z "${pr}" ]; then
        echo "ERROR: no PR found for branch '${branch}'. Run 'just pr' first, then 'just bench-report'." >&2
        exit 1
    fi
    # Derive artifact slug from branch name.
    slug="pr-${pr}-${branch}"
    mkdir -p bench-results
    # Snapshot the run's db into the branch under the final name.
    src="{{_bench_output}}/benchmark.db"
    ssh {{_thor}} "docker exec {{_ctr}} bash -c 'test -f ${src} && cp ${src} {{_wdir}}/bench-results/${slug}.db' || { echo 'ERROR: no benchmark.db — run just bench first'; exit 1; }"
    # Fetch the baseline from origin/main. On a fresh repo there may be none;
    # in that case we self-compare (baseline == current) so the artifacts are
    # generated and the PR still gets chart + table + validation block. The
    # %-deltas are 0 by definition and the verdict is MAYBE, but correctness
    # checking still works — a self-compare with bad pgx output still trips
    # the NAY gate.
    git fetch origin main --quiet
    # git ls-tree pathspec doesn't expand shell-style wildcards — the quoted
    # 'bench-results/pr-*.db' was always returning empty, silently forcing the
    # bootstrap branch even when a real baseline existed. List the whole
    # bench-results/ dir and grep instead. `sort -V` picks highest pr-N
    # numerically — de facto "most recent merge" in FIFO-merge order.
    baseline_path=$(git ls-tree -r --name-only origin/main -- bench-results/ 2>/dev/null | grep -E '^bench-results/pr-.*\.db$' | sort -V | tail -1 || true)
    if [ -z "${baseline_path}" ]; then
        echo "NOTE: no baseline on origin/main:bench-results/ — self-comparing. This PR will seed the baseline for future PRs." >&2
        baseline_name="bootstrap-self.db"
        ssh {{_thor}} "docker exec {{_ctr}} cp {{_wdir}}/bench-results/${slug}.db /tmp/${baseline_name}"
    else
        baseline_name=$(basename "${baseline_path}")
        # Stage the baseline on thor (not committed — just in /tmp for report.py).
        git show "origin/main:${baseline_path}" > /tmp/${baseline_name}
        scp -q /tmp/${baseline_name} {{_thor}}:/tmp/${baseline_name}
        rm /tmp/${baseline_name}
        ssh {{_thor}} "docker cp /tmp/${baseline_name} {{_ctr}}:/tmp/${baseline_name}"
    fi
    ssh {{_thor}} "docker exec {{_ctr}} python3 {{_wdir}}/benchmark/report.py \
        --baseline /tmp/${baseline_name} \
        --current {{_wdir}}/bench-results/${slug}.db \
        --out {{_wdir}}/bench-results/${slug} \
        --chart-url \"https://raw.githubusercontent.com/zyros-dev/pgx-lower/${branch}/bench-results/${slug}.png\""
    echo ""
    echo "Artifacts: bench-results/${slug}.{db,png,md}"
    echo "Baseline : ${baseline_name} (from origin/main)"
    # Force mutagen to finish syncing thor→mac before we try to read the
    # .md that report.py just wrote. Without this, the Python replace step
    # below races: report.py finishes on thor, we immediately try to open
    # bench-results/<slug>.md locally, and mutagen hasn't caught up yet.
    # `mutagen sync flush` blocks until the cycle completes.
    mutagen sync flush "pgx-lower" >/dev/null 2>&1 || true
    # Auto-inject the .md into the PR body, replacing the stats-summary
    # placeholder that `just pr` left. The agent still fills in Summary by
    # hand; everything else is assembled. Safe to re-run — idempotent
    # because we only replace the literal placeholder string (if absent,
    # nothing happens).
    current_body=$(gh pr view "${pr}" --json body -q .body)
    placeholder="<paste the stats summary block here — required>"
    if printf '%s' "${current_body}" | grep -qF "${placeholder}"; then
        # Use python for the replacement so bench report content isn't
        # subject to sed's metachar quirks.
        new_body=$(printf '%s' "${current_body}" | python3 -c "import sys, pathlib; body = sys.stdin.read(); md = pathlib.Path('bench-results/${slug}.md').read_text(); print(body.replace('${placeholder}', md), end='')")
        gh pr edit "${pr}" --body "${new_body}" >/dev/null
        echo "PR  body  : injected bench report block into PR #${pr}."
        # Detect remaining template placeholders and flag them explicitly —
        # the old "PR body updated" message overstated what happened and left
        # agents thinking they were done when the Summary section was still a
        # literal "<what and why>".
        remaining=$(gh pr view "${pr}" --json body -q .body | grep -oE '<[^>]*>' | sort -u | grep -v '<br' || true)
        if [ -n "${remaining}" ]; then
            echo "NOTE      : PR body still has unfilled template placeholders:"
            printf '            %s\n' ${remaining}
            echo "            Fill them in with \`gh pr edit ${pr} --body ...\` before requesting review."
        fi
    else
        echo "PR  body  : placeholder already replaced — skipping auto-inject. Paste bench-results/${slug}.md manually if needed."
    fi

# --- Queue ops ------------------------------------------------------------

# Show the build queue (pending + running + recent finished jobs) with a
# one-line summary up top so you know at a glance whether your job will wait.
# "deep" per the skill = 3+ queued/running jobs ahead of you; at that point,
# expect noticeable wait. Count lines parse tsp's table format, not tsp -l
# machine output, so slight future format drift is OK.
queue:
    #!/usr/bin/env bash
    set -o pipefail
    for q in {{_build_q}} {{_check_q}}; do
        echo "=== ${q} queue ==="
        out=$(ssh {{_thor}} "TS_SOCKET=/tmp/${q}.sock tsp" 2>&1 || true)
        # Count data rows (skip the "ID State Output..." header).
        jobs=$(printf '%s\n' "${out}" | awk 'NR>1 && NF>0' | wc -l | tr -d ' ')
        running=$(printf '%s\n' "${out}" | awk '/running/' | wc -l | tr -d ' ')
        queued=$(printf '%s\n' "${out}" | awk '/queued/' | wc -l | tr -d ' ')
        echo "  ${jobs} total  (${running} running, ${queued} queued)"
        printf '%s\n' "${out}"
    done

# Tail the live output of a running job from the build queue.
tail ID:
    ssh {{_thor}} 'TS_SOCKET=/tmp/{{_build_q}}.sock tsp -t {{ID}}'

# Cancel and remove a queued or running build job.
cancel ID:
    ssh {{_thor}} 'TS_SOCKET=/tmp/{{_build_q}}.sock tsp -k {{ID}} || true; TS_SOCKET=/tmp/{{_build_q}}.sock tsp -r {{ID}}'

# --- Sync ------------------------------------------------------------------

# Terminate + recreate the main-repo mutagen session (name `pgx-lower`)
# with the canonical ignore list. Use this when the session's ignore list
# has drifted — e.g. a missing `/build-artifacts/` ignore lets run.py's
# root-owned sqlite files sync back to mac and trip "attempt to write a
# readonly database" on the next bench. mutagen has no in-place ignore
# editor, so the recipe is a full terminate + recreate. Safe to run at any
# time; mutagen reconciles state on the next scan.
sync-main-reset:
    #!/usr/bin/env bash
    set -euo pipefail
    mutagen sync terminate pgx-lower >/dev/null 2>&1 || true
    mutagen sync create \
        --name=pgx-lower \
        --sync-mode=two-way-resolved \
        --ignore='/build-artifacts/' --ignore='/build-*/' --ignore='/build-docker-*/' --ignore='/postgres-debug/' \
        --ignore='/.worktrees/' \
        --ignore='__pycache__/' --ignore='*.pyc' --ignore='*.tar.gz' \
        --ignore='/.venv/' --ignore='/.idea/' --ignore='/.vscode/' \
        --ignore='/benchmark_results/' \
        "{{_main_root}}" {{_thor}}:/home/zel/repos/pgx-lower
    echo "sync-main-reset: main session recreated with canonical ignores."

# --- PR --------------------------------------------------------------------

# Open a PR from the current branch against main. Second arg is an
# optional Summary body (replaces the `<what and why>` placeholder), so
# agents that know their summary at PR-open time skip a trailing
# `gh pr edit` pass. `just bench-report` handles the stats-summary
# placeholder later.
#
# Summary resolution, in order:
#   1. SUMMARY arg, if passed and not the placeholder default.
#   2. BODY_FILE, if set (env var), read from that path.
#   3. The last commit's body (everything after the subject line), if non-empty.
#   4. Fall back to the literal `<what and why>` placeholder so agents
#      who forgot to fill it in still get a functional PR (and `bench-report`
#      warns them at the end).
#
#   just pr "spec 03: plan-shape compile cache"
#   just pr "fix bench race" "bench-report was reading .md before sync."
#   BODY_FILE=/tmp/pr-body.md just pr "fix bench race"
pr TITLE SUMMARY='<what and why>':
    #!/usr/bin/env bash
    set -euo pipefail
    summary="{{SUMMARY}}"
    # Track which source the Summary body came from so we can print an
    # operator-facing diagnostic at the end. Saves the caller a follow-
    # up `gh pr view` to confirm whether `<what and why>` was replaced.
    source="arg"
    if [ "${summary}" = '<what and why>' ]; then
        source="placeholder"
        if [ -n "${BODY_FILE:-}" ] && [ -r "${BODY_FILE}" ]; then
            summary=$(cat "${BODY_FILE}")
            source="BODY_FILE"
        else
            # Pull the body of the last commit (%b = body only, no subject).
            # Drop any trailing Co-Authored-By trailer block — that's a
            # machine-authored trailer, not part of the Summary prose. Then
            # strip leading/trailing blank lines. If the commit is
            # subject-only, %b is empty and we fall through to the
            # placeholder default.
            commit_body=$(git log -1 --pretty=%b | awk '/^Co-Authored-By:/{exit} {print}')
            # Trim leading blank lines.
            commit_body=$(printf '%s' "${commit_body}" | awk 'NF{found=1} found')
            # Trim trailing blank lines.
            commit_body=$(printf '%s' "${commit_body}" | awk '{a[NR]=$0} END {last=NR; while(last>0 && a[last]=="") last--; for(i=1;i<=last;i++) print a[i]}')
            if [ -n "${commit_body}" ]; then
                summary="${commit_body}"
                source="commit body"
            fi
        fi
    fi
    url=$(gh pr create --base main --head "$(git rev-parse --abbrev-ref HEAD)" --title "{{TITLE}}" --body "$(printf '## Summary\n\n%s\n\n<paste the stats summary block here — required>\n\n## Test plan\n- [ ] just check\n- [ ] just test\n- [ ] just bench\n' "${summary}")")
    # Diagnostic block: tells the operator at a glance whether the
    # Summary body was populated from the expected source and whether
    # the `<what and why>` placeholder is still sitting in the PR body
    # waiting for manual replacement. Previously `just pr` only echoed
    # the PR URL, forcing a follow-up `gh pr view` to confirm the
    # state — these three lines eliminate that round-trip.
    pr_num=$(printf '%s' "${url}" | grep -oE '[0-9]+$' | tail -1)
    # The `<what and why>` placeholder is the DEFAULT summary value; it
    # only survives to the rendered PR body when `source == placeholder`
    # (i.e. nothing else populated the Summary). Derive the diagnostic
    # from `source` rather than grepping the summary text itself, to
    # avoid false positives when the summary legitimately discusses the
    # placeholder by name (e.g. a commit body describing this very
    # recipe).
    if [ "${source}" = "placeholder" ]; then
        placeholder_present="yes"
    else
        placeholder_present="no"
    fi
    echo "PR #${pr_num} opened"
    echo "Summary resolved from: ${source}"
    echo "<what and why> still in body: ${placeholder_present}"
    echo "${url}"

# Replace the `<what and why>` Summary placeholder on the current branch's
# open PR in a single gh-pr-edit call. Idempotent — if the placeholder is
# already gone, this is a no-op (and says so). Use this after `just pr` +
# `just bench-report` when you're ready to commit your Summary text, so you
# don't have to hand-craft a full gh pr edit --body invocation.
#
#   just pr-summary "Fixes bench race: bench-report was reading .md before sync flushed thor→mac."
pr-summary SUMMARY:
    #!/usr/bin/env bash
    set -euo pipefail
    pr=$(gh pr view --json number -q .number 2>/dev/null || true)
    if [ -z "${pr}" ]; then
        echo "ERROR: no open PR on this branch. Run 'just pr' first." >&2
        exit 1
    fi
    body=$(gh pr view "${pr}" --json body -q .body)
    placeholder='<what and why>'
    if ! printf '%s' "${body}" | grep -qF "${placeholder}"; then
        echo "pr-summary: placeholder already replaced on PR #${pr}. No-op."
        exit 0
    fi
    # Pass SUMMARY via an env var so shell-special chars (quotes, backticks,
    # backslashes) in the user's text don't break the substitution.
    export _PR_SUMMARY_TEXT="{{SUMMARY}}"
    new_body=$(printf '%s' "${body}" | python3 -c "import os, sys; body = sys.stdin.read(); print(body.replace('<what and why>', os.environ['_PR_SUMMARY_TEXT']), end='')")
    gh pr edit "${pr}" --body "${new_body}" >/dev/null
    echo "pr-summary: replaced <what and why> on PR #${pr}."
