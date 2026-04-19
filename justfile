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
_bdir := _wdir + "/build-docker-ptest"

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
    @branch=$(git rev-parse --abbrev-ref HEAD); session="pgx-lower-${branch}"; [ "${branch}" = "main" ] && session="pgx-lower"; mutagen sync flush "${session}" >/dev/null 2>&1 || true

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
    # just compile" cargo-cult. If we're on main the session is
    # "pgx-lower" (no branch suffix); on a worktree it's "pgx-lower-<branch>".
    branch=$(git rev-parse --abbrev-ref HEAD)
    session="pgx-lower-${branch}"
    [ "${branch}" = "main" ] && session="pgx-lower"
    mutagen sync flush "${session}" >/dev/null 2>&1 || true
    ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "mkdir -p {{_bdir}} && cd {{_bdir}} && ([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache {{_wdir}}) && cmake --build . && cmake --install .") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id' 2>&1 | tee /tmp/pgx-compile.out
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        ninja_targets=$(grep -cE '^\[[0-9]+/[0-9]+\]' /tmp/pgx-compile.out 2>/dev/null || echo 0)
        echo ""
        echo "BUILD OK — ${ninja_targets} ninja step(s), pgx_lower.so installed"
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
# recipe reads build-docker-ptest/extension/results/<name>.out on thor.
expected-from-results TEST:
    #!/usr/bin/env bash
    set -euo pipefail
    src="{{_bdir}}/extension/results/{{TEST}}.out"
    if ! ssh {{_thor}} "docker exec {{_ctr}} test -f ${src}"; then
        echo "ERROR: ${src} not found on thor. Run 'just test' first so pg_regress generates the result." >&2
        exit 1
    fi
    mkdir -p tests/expected
    ssh {{_thor}} "docker exec {{_ctr}} cat ${src}" > "tests/expected/{{TEST}}.out"
    echo "Wrote tests/expected/{{TEST}}.out ($(wc -l <tests/expected/{{TEST}}.out) lines, $(wc -c <tests/expected/{{TEST}}.out) bytes)."
    echo "Re-run 'just test' — it should now pass for this case."

# Unit tests (gtest) — the primary TDD home for most spec work. Builds
# a separate build-docker-utest/ dir with -DBUILDING_UNIT_TESTS=ON and
# runs ctest. Configures on first invocation; subsequent runs are
# incremental ninja + ccache. Queued on the build queue so it doesn't
# race compile/test/bench.
utest: _preflight
    #!/usr/bin/env bash
    set -o pipefail
    ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "mkdir -p {{_wdir}}/build-docker-utest && cd {{_wdir}}/build-docker-utest && ([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILDING_UNIT_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache {{_wdir}}) && cmake --build . && ctest --output-on-failure") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id' 2>&1 | tee /tmp/pgx-utest.out
    rc=${PIPESTATUS[0]}
    passed=$(grep -oE '^[0-9]+% tests passed' /tmp/pgx-utest.out | tail -1 || echo "unknown")
    failed=$(grep -oE '[0-9]+ tests? failed out of [0-9]+' /tmp/pgx-utest.out | tail -1 || echo "")
    echo ""
    if [ "$rc" -eq 0 ]; then
        echo "UTEST OK — ${passed}"
    else
        echo "UTEST FAILED — ${failed:-unknown}, exit $rc. Last 30 lines:"
        tail -n 30 /tmp/pgx-utest.out
        exit "$rc"
    fi

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
# spec work, prefer `just utest` (faster, scoped to the thing you're
# actually changing). See SKILL.md step 2.
test: _preflight
    #!/usr/bin/env bash
    set -euo pipefail
    ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "mkdir -p {{_bdir}} && cd {{_bdir}} && ([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache {{_wdir}}) && cmake --build . && cmake --install . && mkdir -p /tmp/pgx_ir && chmod 777 /tmp/pgx_ir; chmod o+x /workspace/.worktrees 2>/dev/null || true; chmod -R o+rX {{_wdir}}; chown -R postgres:postgres {{_bdir}} && cd {{_bdir}} && (su postgres -c \"ctest -V\" 2>&1 | tee /tmp/ctest.out; cat /tmp/ctest.out | python3 {{_wdir}}/scripts/ptest_with_baseline.py --baseline-file {{_wdir}}/tests/pg_regress_baseline.txt)") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

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
# Recreates benchmark/output/ each run — a partial interrupted earlier run
# can leave sqlite journal/lock state that makes subsequent connects open
# readonly and bomb mid-run with "attempt to write a readonly database".
bench: _preflight
    @ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "rm -rf {{_wdir}}/benchmark/output && mkdir -p {{_wdir}}/benchmark/output && chmod 777 {{_wdir}}/benchmark/output && cd {{_wdir}} && python3 benchmark/tpch/run.py 0.5 --port 5432 --container {{_ctr}} --indexes --skip q17,q20 --iterations 1") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

# Deeper-signal benchmark: SF=1, 1 iteration. ~10 min first time, ~6 min
# cached. Run before merging anything that claims a performance improvement
# where SF=0.5's numbers feel marginal. At SF=1 the per-query wall time is
# long enough (seconds) that ±5% is real signal.
bench-merge: _preflight
    @ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "rm -rf {{_wdir}}/benchmark/output && mkdir -p {{_wdir}}/benchmark/output && chmod 777 {{_wdir}}/benchmark/output && cd {{_wdir}} && python3 benchmark/tpch/run.py 1.0 --port 5432 --container {{_ctr}} --indexes --skip q17,q20 --iterations 1") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

# Generate the PR benchmark report. Requires an open PR (the PR number
# becomes part of the filename). Snapshots the current benchmark.db to
# ./benchmarks/pr-<N>-spec-<NN>-<slug>.db (or pr-<N>-<slug>.db for
# non-spec branches), pulls the baseline db directly from origin/main (not
# committed to feature branches), and emits matching .png + .md.
#
# Naming rules:
#   spec branch   (spec-NN-<slug>) → pr-<N>-spec-<NN>-<slug>.db
#   non-spec      (<slug>)         → pr-<N>-<slug>.db
#
# Baseline: the alphanumerically latest .db in origin/main:benchmarks/
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
    if [[ "${branch}" =~ ^spec-([0-9]+)-(.+)$ ]]; then
        slug="pr-${pr}-spec-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
    else
        slug="pr-${pr}-${branch}"
    fi
    mkdir -p benchmarks
    # Snapshot the run's db into the branch under the final name.
    src="{{_wdir}}/benchmark/output/benchmark.db"
    ssh {{_thor}} "docker exec {{_ctr}} bash -c 'test -f ${src} && cp ${src} {{_wdir}}/benchmarks/${slug}.db' || { echo 'ERROR: no benchmark.db — run just bench first'; exit 1; }"
    # Fetch the baseline from origin/main. On a fresh repo there may be none;
    # in that case we self-compare (baseline == current) so the artifacts are
    # generated and the PR still gets chart + table + validation block. The
    # %-deltas are 0 by definition and the verdict is MAYBE, but correctness
    # checking still works — a self-compare with bad pgx output still trips
    # the NAY gate.
    git fetch origin main --quiet
    # git ls-tree pathspec doesn't expand shell-style wildcards — the quoted
    # 'benchmarks/pr-*.db' was always returning empty, silently forcing the
    # bootstrap branch even when a real baseline existed. List the whole
    # benchmarks/ dir and grep instead. `sort -V` picks highest pr-N
    # numerically — de facto "most recent merge" in FIFO-merge order.
    baseline_path=$(git ls-tree -r --name-only origin/main -- benchmarks/ 2>/dev/null | grep -E '^benchmarks/pr-.*\.db$' | sort -V | tail -1 || true)
    if [ -z "${baseline_path}" ]; then
        echo "NOTE: no baseline on origin/main:benchmarks/ — self-comparing. This PR will seed the baseline for future PRs." >&2
        baseline_name="bootstrap-self.db"
        ssh {{_thor}} "docker exec {{_ctr}} cp {{_wdir}}/benchmarks/${slug}.db /tmp/${baseline_name}"
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
        --current {{_wdir}}/benchmarks/${slug}.db \
        --out {{_wdir}}/benchmarks/${slug} \
        --chart-url \"https://raw.githubusercontent.com/zyros-dev/pgx-lower/${branch}/benchmarks/${slug}.png\""
    echo ""
    echo "Artifacts: benchmarks/${slug}.{db,png,md}"
    echo "Baseline : ${baseline_name} (from origin/main)"
    # Force mutagen to finish syncing thor→mac before we try to read the
    # .md that report.py just wrote. Without this, the Python replace step
    # below races: report.py finishes on thor, we immediately try to open
    # benchmarks/<slug>.md locally, and mutagen hasn't caught up yet.
    # `mutagen sync flush` blocks until the cycle completes.
    session="pgx-lower-${branch}"
    # If we're on main, the session name is "pgx-lower" (no branch suffix).
    if [ "${branch}" = "main" ]; then session="pgx-lower"; fi
    mutagen sync flush "${session}" >/dev/null 2>&1 || true
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
        new_body=$(printf '%s' "${current_body}" | python3 -c "import sys, pathlib; body = sys.stdin.read(); md = pathlib.Path('benchmarks/${slug}.md').read_text(); print(body.replace('${placeholder}', md), end='')")
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
        echo "PR  body  : placeholder already replaced — skipping auto-inject. Paste benchmarks/${slug}.md manually if needed."
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

# --- Worktree lifecycle ----------------------------------------------------

# Create a git worktree on mac + thor and a mutagen sync session between them.
# Usage: just worktree-new feat-foo
#
# Fully retry-idempotent: if a previous attempt aborted partway and left
# detritus behind (a `.worktrees/<NAME>/` directory git no longer tracks,
# a local `<NAME>` branch without a matching worktree, a thor-side
# `.worktrees/<NAME>/` left by container-root writes, etc.), this recipe
# cleans it up and proceeds. The clean-ops are wrapped in `|| true` so
# a fresh invocation with nothing to clean no-ops through them. The
# thor-side cleanup uses `docker exec` first (to delete container-root-
# owned build artifacts) then a plain ssh rm -rf for anything else.
# Mirrors the same rationale as `worktree-rm`'s cleanup comments.
#
# Rationale for not just erroring out: earlier versions refused to
# reuse a slug when the local branch existed, which left agents having
# to hand-run `rm -rf` + `git branch -D` to recover from a failed
# worktree-new. `spec-abandon` wasn't the right tool either when the
# spec had already been re-claimed — running abandon would undo the
# claim. Making worktree-new self-healing removes the manual-recovery
# path entirely: you re-run the recipe and it converges.
worktree-new NAME:
    #!/usr/bin/env bash
    set -euo pipefail
    test -n "{{NAME}}" || { echo "NAME required"; exit 1; }
    # --- preflight cleanup (idempotent) ------------------------------
    # Prune stale worktree bookkeeping entries first. If a previous
    # .worktrees/<NAME>/ was deleted behind git's back (manual rm -rf,
    # crashed `git worktree add`), git still thinks it owns that path
    # and will refuse `add` with "already exists". `prune` clears it.
    git worktree prune 2>/dev/null || true
    # Remove any leftover directory on mac. `git worktree add` hard-
    # fails with "'.worktrees/<NAME>' already exists" if the path
    # exists on disk but isn't a registered worktree.
    rm -rf ".worktrees/{{NAME}}" 2>/dev/null || true
    # Delete any dangling local branch. Without this, `git worktree add
    # -b` fails with "a branch named '<NAME>' already exists". -D
    # (force) is intentional: if we're here, either the branch was
    # never pushed (safe to drop) or it was pushed and the caller knows
    # they're throwing it away to restart.
    git branch -D "{{NAME}}" 2>/dev/null || true
    # Same cleanup on thor: prune its worktree bookkeeping, then
    # delete any leftover .worktrees/<NAME>/ directory. Container-root-
    # owned files inside build-docker-*/ subdirs can't be removed by
    # the ssh user (uid 1000), so do the rm inside the container as
    # root first; anything left over gets swept by the outer ssh rm.
    # All four ops tolerate missing state.
    ssh {{_thor}} 'cd ~/repos/pgx-lower && git worktree prune 2>/dev/null || true'
    ssh {{_thor}} 'docker exec {{_ctr}} rm -rf /workspace/.worktrees/{{NAME}} 2>/dev/null || true'
    ssh {{_thor}} 'rm -rf ~/repos/pgx-lower/.worktrees/{{NAME}} 2>/dev/null || true'
    # --- create fresh state ------------------------------------------
    git fetch origin main --quiet
    git worktree add -b "{{NAME}}" ".worktrees/{{NAME}}" origin/main
    ssh {{_thor}} 'cd ~/repos/pgx-lower && git fetch origin && git worktree add .worktrees/{{NAME}} 2>/dev/null || true'
    # /benchmark/output/ must be in this ignore list. run.py on thor writes
    # sqlite files there as docker root; without the ignore, mutagen syncs
    # them back to mac with altered ownership mid-run, and subsequent writes
    # hit "attempt to write a readonly database" partway through. This
    # bit the canary (see round 7). Keep it aligned with `sync-main-reset`.
    mutagen sync create \
        --name=pgx-lower-{{NAME}} \
        --sync-mode=two-way-resolved \
        --ignore='/build-*/' --ignore='/build-docker-*/' --ignore='/postgres-debug/' \
        --ignore='__pycache__/' --ignore='*.pyc' --ignore='*.tar.gz' \
        --ignore='/.venv/' --ignore='/.idea/' --ignore='/.vscode/' \
        --ignore='/benchmark/output/' \
        .worktrees/{{NAME}} {{_thor}}:/home/zel/repos/pgx-lower/.worktrees/{{NAME}}
    echo ""
    echo "Worktree ready. On mac: cd .worktrees/{{NAME}}"
    echo "Mutagen session: pgx-lower-{{NAME}}"

# Tear down a worktree and its sync. Fully idempotent — every step is
# allowed to fail (missing session, already-removed worktree, etc.)
# because we want repeated calls and partial-state recovery to both
# converge on "nothing here."
#
# After git-level removal, we unconditionally `rm -rf` the mac-side
# .worktrees/<name>/ directory and the thor-side equivalent. `git
# worktree remove --force` does NOT delete the dir if git no longer
# recognizes it as a working tree (e.g. it was already pruned, or
# previous cleanup pass lost the bookkeeping) — it bails with "not a
# working tree" and leaves a full tree on disk. That residual tree is
# a landmine: a subsequent `just worktree-new <same-name>` fails with
# "path already exists," and a subsequent mutagen create sees stale
# files that don't correspond to any branch. Removing it on the way
# out guarantees re-creation works without manual `rm -rf`.
#
# The thor-side rm is done TWICE, in this order: first via `docker
# exec` as container-root (uid 0), then via plain ssh as zel (uid
# 1000). build-docker-ptest/ and build-docker-utest/ contain files
# chowned to `postgres:postgres` by `cmake --install` + ctest inside
# the container — the outer ssh rm can't unlink those and fails with
# "Permission denied". The inner docker rm nukes the whole .worktrees/
# <NAME>/ including the container-owned files; the outer rm mops up
# anything the container didn't see (e.g. if the container had been
# stopped or the mount wasn't active). Either alone may leave residue
# depending on runtime state, so we run both.
worktree-rm NAME:
    -mutagen sync terminate pgx-lower-{{NAME}}
    -ssh {{_thor}} 'cd ~/repos/pgx-lower && git worktree remove --force .worktrees/{{NAME}}'
    -git worktree remove --force .worktrees/{{NAME}}
    -rm -rf .worktrees/{{NAME}}
    -ssh {{_thor}} 'docker exec {{_ctr}} rm -rf /workspace/.worktrees/{{NAME}} 2>/dev/null || true'
    -ssh {{_thor}} 'rm -rf ~/repos/pgx-lower/.worktrees/{{NAME}}'

# Terminate + recreate the main-repo mutagen session (name `pgx-lower`)
# with the canonical ignore list. Use this when the main session's
# ignore list has drifted from what `just worktree-new` creates for
# worktrees — e.g. a missing `/benchmark/output/` ignore (the cause of
# the "attempt to write a readonly database" bench failure that
# prompted round 7). mutagen has no in-place ignore editor, so the
# recipe is a full terminate + recreate. Safe to run at any time;
# mutagen reconciles state on the next scan.
sync-main-reset:
    #!/usr/bin/env bash
    set -euo pipefail
    mutagen sync terminate pgx-lower >/dev/null 2>&1 || true
    mutagen sync create \
        --name=pgx-lower \
        --sync-mode=two-way-resolved \
        --ignore='/build-*/' --ignore='/build-docker-*/' --ignore='/postgres-debug/' \
        --ignore='__pycache__/' --ignore='*.pyc' --ignore='*.tar.gz' \
        --ignore='/.venv/' --ignore='/.idea/' --ignore='/.vscode/' \
        --ignore='/benchmark/output/' --ignore='/benchmark_results/' \
        "{{_main_root}}" {{_thor}}:/home/zel/repos/pgx-lower
    echo "sync-main-reset: main session recreated with canonical ignores."

# Sweep dangling state from prior worktrees that merged or got abandoned
# without a clean `just worktree-rm`: terminate mutagen sessions whose
# local worktree dir is gone, prune thor-side git worktrees that show as
# prunable, prune mac-side. Idempotent — safe to run whenever
# `just worktree-list` shows noise.
worktree-sweep:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "=== pruning git worktrees (mac) ==="
    git worktree prune -v
    echo ""
    echo "=== pruning git worktrees (thor) ==="
    ssh {{_thor}} 'cd ~/repos/pgx-lower && git worktree prune -v'
    echo ""
    echo "=== terminating stale mutagen sessions ==="
    # For each session named pgx-lower-<slug>, if .worktrees/<slug> is gone
    # locally, terminate it.
    for name in $(mutagen sync list 2>/dev/null | awk -F': ' '/^Name:/ {print $2}' | grep '^pgx-lower-' || true); do
        slug="${name#pgx-lower-}"
        if [ ! -d ".worktrees/${slug}" ]; then
            echo "  terminating ${name} (worktree .worktrees/${slug} is gone)"
            mutagen sync terminate "${name}" >/dev/null 2>&1 || true
        fi
    done
    echo ""
    echo "Done."

# List worktrees + their sync sessions, auto-pruning stale entries first so
# thor-side rows labeled "prunable" don't leak in as live worktrees on the
# mac side (and vice versa). Live and prunable are split visually so agents
# can see at a glance which worktrees are actually on-disk.
worktree-list:
    #!/usr/bin/env bash
    set -o pipefail
    # Prune first — this clears out branches whose working-tree dir has been
    # deleted behind git's back (common after mergers + manual rm -rf).
    # `--dry-run` would merely warn; we actively prune because the next
    # `git worktree list` on mac or thor would otherwise flag the same rows
    # as "prunable" again on subsequent calls.
    git worktree prune >/dev/null 2>&1 || true
    ssh {{_thor}} 'cd ~/repos/pgx-lower && git worktree prune >/dev/null 2>&1' || true
    _split_list() {
        local label="$1"; shift
        local raw="$1"
        # Column layout from `git worktree list` is path/commit/branch; the
        # "prunable" marker, when present, shows up on its own row (one per
        # worktree). We walk the porcelain form which is unambiguous.
        # Fall back to plain output if the porcelain variant isn't available.
        echo "--- ${label} (live) ---"
        printf '%s\n' "${raw}" | grep -v 'prunable' || true
        local pruned=$(printf '%s\n' "${raw}" | grep 'prunable' || true)
        if [ -n "${pruned}" ]; then
            echo "--- ${label} (prunable) ---"
            printf '%s\n' "${pruned}"
        fi
    }
    echo "=== mac worktrees ==="
    mac_raw=$(git worktree list --porcelain 2>/dev/null | awk '
        /^worktree /{wt=$2}
        /^HEAD /{head=$2}
        /^branch /{branch=$2}
        /^prunable/{pr=1}
        /^$/{if(wt){printf "%s  %s  %s%s\n", wt, substr(head,1,7), branch, (pr?"  [prunable]":""); wt=""; head=""; branch=""; pr=0}}
        END{if(wt){printf "%s  %s  %s%s\n", wt, substr(head,1,7), branch, (pr?"  [prunable]":"")}}')
    _split_list "mac" "${mac_raw}"
    echo ""
    echo "=== thor worktrees ==="
    thor_raw=$(ssh {{_thor}} 'cd ~/repos/pgx-lower && git worktree list --porcelain' 2>/dev/null | awk '
        /^worktree /{wt=$2}
        /^HEAD /{head=$2}
        /^branch /{branch=$2}
        /^prunable/{pr=1}
        /^$/{if(wt){printf "%s  %s  %s%s\n", wt, substr(head,1,7), branch, (pr?"  [prunable]":""); wt=""; head=""; branch=""; pr=0}}
        END{if(wt){printf "%s  %s  %s%s\n", wt, substr(head,1,7), branch, (pr?"  [prunable]":"")}}')
    _split_list "thor" "${thor_raw}"
    echo ""
    echo "=== mutagen sessions ==="
    mutagen sync list | grep -E '^Name:|^Status:' || true
    echo ""
    # Orphan detection: a mutagen session named pgx-lower-<slug> with no
    # matching .worktrees/<slug>/ directory on mac is a zombie left over
    # from an abandoned spec or a crashed worktree-rm. It still watches
    # and syncs, which is how the benchmark/output/ readonly-db trap
    # pops up silently on the *next* worktree's first bench. Surface
    # these here so `just worktree-list` doubles as a doctor pass.
    echo "=== orphaned mutagen sessions ==="
    # Resolve to the main repo root so the .worktrees/ check works regardless
    # of whether `just worktree-list` was invoked from the main checkout or
    # from inside a worktree (where `.worktrees/` doesn't exist and every
    # session would otherwise be misflagged as orphaned).
    main_root="{{_main_root}}"
    orphans=""
    for name in $(mutagen sync list 2>/dev/null | awk -F': ' '/^Name:/ {print $2}' | grep -E '^pgx-lower-' || true); do
        slug="${name#pgx-lower-}"
        # The bare "pgx-lower" session maps to the main checkout, not a worktree.
        if [ "${name}" = "pgx-lower" ]; then continue; fi
        if [ ! -d "${main_root}/.worktrees/${slug}" ]; then
            orphans+="  ${name}  (no .worktrees/${slug}/ on mac)"$'\n'
        fi
    done
    if [ -n "${orphans}" ]; then
        printf '%s' "${orphans}"
        echo ""
        echo "To clean up: \`just worktree-sweep\` terminates all of them in one pass,"
        echo "or \`mutagen sync terminate <name>\` for a specific session."
    else
        echo "  (none)"
    fi

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

# --- Spec coordination ----------------------------------------------------
# These wrap edits to specs/STATUS.md and commit atomically against main so
# concurrent agents can't claim the same spec. Run from the main checkout
# (not a worktree) — the helper script refuses otherwise.

# Show the spec board.
spec-status:
    @cat specs/STATUS.md

# Mark a spec as in_progress. NN is the spec number (e.g. 03), BRANCH is the
# worktree name you'll use. Owner defaults to <git user.email username>-MMDD;
# override via OWNER=...
# Usage: just spec-claim 03 spec-03-cache
#
# Note on preflight: earlier iterations considered adding a check here
# for local detritus (`.worktrees/<BRANCH>/` already on disk, local
# branch already exists) so the claim wouldn't succeed-then-leave-you-
# in-limbo if the follow-up `just worktree-new` failed. That preflight
# is intentionally NOT done here; it lives in `worktree-new`, which is
# now fully retry-idempotent: if you re-run it after a prior aborted
# attempt, it prunes git's stale bookkeeping, removes leftover
# directories on mac + thor (including container-root-owned files via
# `docker exec`), and deletes dangling local branches before creating
# fresh state. Net result: a failed claim→worktree-new sequence is
# recoverable by simply re-running `just worktree-new <BRANCH>` — no
# need to `spec-abandon` (which would undo the claim) and no need to
# hand-run `rm -rf` + `git branch -D`. Keeping spec-claim as a pure
# STATUS.md mutation means the two concerns stay separable: the claim
# row reflects board state, worktree-new owns filesystem state.
spec-claim NN BRANCH OWNER='':
    OWNER="{{OWNER}}" python3 scripts/spec_status.py claim "{{NN}}" "{{BRANCH}}"

# Release a claim (e.g. abandoning the work). Sets state back to available.
# Use when the claim is still "just a row on the board" — no branch pushed,
# no PR open, no worktree on disk. For in-between messes (in_review with a
# closed PR, zombie worktree, dangling remote branch), use `spec-abandon` —
# it does the full tear-down, including closing the PR and renaming it so
# future agents don't resurrect it.
spec-release NN:
    python3 scripts/spec_status.py release "{{NN}}"

# Full tear-down recovery path for a zombie in_progress / in_review spec.
# Closes the PR (if open), renames it so it doesn't look implement-able,
# deletes the remote branch, removes the worktree on mac + thor, terminates
# the mutagen session, and flips the STATUS row back to available. Idempotent
# — missing pieces just skip, it never hard-fails on "already gone".
#
# Usage: just spec-abandon NN "<reason>"
spec-abandon NN REASON:
    #!/usr/bin/env bash
    set -uo pipefail
    branch=$(python3 scripts/spec_status.py read_branch "{{NN}}" 2>/dev/null || true)
    if [ -z "$branch" ]; then
        echo "Spec {{NN}} has no branch recorded on STATUS; releasing claim only."
        python3 scripts/spec_status.py release "{{NN}}"
        exit 0
    fi
    echo "Tearing down spec {{NN}} (branch: ${branch})"
    # Close the PR if one exists, and rewrite its title so future agents
    # don't re-open it thinking it's implementable work.
    pr=$(gh pr list --head "${branch}" --state all --json number,state -q '.[] | select(.state=="OPEN") | .number' | head -1)
    if [ -n "$pr" ]; then
        gh pr edit "$pr" --title "[abandoned] spec {{NN}} — {{REASON}}" >/dev/null 2>&1 || true
        gh pr close "$pr" --delete-branch --comment "Abandoned: {{REASON}}. Do not re-open — use \`just spec-claim {{NN}} <new-branch>\` to start fresh." >/dev/null 2>&1 || true
        echo "  PR #${pr} closed + renamed."
    else
        # Also rewrite any already-closed PR's title as a breadcrumb, per the
        # feedback rule: abandoned PRs should read as abandoned so an agent
        # skimming gh pr list doesn't try to resurrect them.
        closed=$(gh pr list --head "${branch}" --state closed --json number,title -q '.[].number' | head -1)
        if [ -n "$closed" ]; then
            gh pr edit "$closed" --title "[abandoned] spec {{NN}} — {{REASON}}" >/dev/null 2>&1 || true
            echo "  Previously-closed PR #${closed} retitled as abandoned."
        fi
    fi
    # Delete remote branch if it still exists.
    git push origin --delete "${branch}" 2>/dev/null && echo "  remote branch ${branch} deleted." || echo "  remote branch ${branch} already gone."
    # Tear down the worktree (does mutagen sync terminate + worktree remove on both sides; all idempotent via `-`).
    just worktree-rm "${branch}" || true
    # Delete the LOCAL branch on the main checkout. If we skip this, a subsequent
    # `just spec-claim NN <same-slug>` + `just worktree-new <same-slug>` will
    # silently reuse the stale branch with the abandoned implementation still
    # committed — surprising and nearly invisible. -D forces deletion even if
    # unmerged (which is correct: we're abandoning).
    git branch -D "${branch}" 2>/dev/null && echo "  local branch ${branch} deleted." || echo "  local branch ${branch} already gone."
    # Flip STATUS.md back to available.
    python3 scripts/spec_status.py release "{{NN}}"
    echo "Spec {{NN}} abandoned and released."

# Mark a spec done after its PR merges. PR is the PR number. Also tears
# down the worktree + mutagen session automatically so those don't pile
# up as "prunable" entries in `just worktree-list` after merges.
spec-complete NN PR:
    #!/usr/bin/env bash
    set -euo pipefail
    branch=$(python3 scripts/spec_status.py read_branch "{{NN}}" 2>/dev/null || true)
    python3 scripts/spec_status.py complete "{{NN}}" "{{PR}}"
    if [ -n "${branch}" ] && [ -d ".worktrees/${branch}" ]; then
        just worktree-rm "${branch}" || true
    fi

# Mark a spec as in_review (PR open). PR is the PR number.
spec-in-review NN PR:
    python3 scripts/spec_status.py in_review "{{NN}}" "{{PR}}"

# Mark a spec as blocked, with a reason note.
spec-block NN REASON:
    python3 scripts/spec_status.py block "{{NN}}" "{{REASON}}"
