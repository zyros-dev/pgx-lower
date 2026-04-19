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

# Verify tsp + dev container are up on thor. Every real recipe depends on this.
_preflight:
    @ssh {{_thor}} 'command -v tsp >/dev/null 2>&1 || { echo "ERROR: task-spooler not installed on thor. Run: ssh comfy sudo apt-get install -y task-spooler"; exit 1; }'
    @ssh {{_thor}} 'docker ps --format "{{{{.Names}}" | grep -q "^{{_ctr}}$" || { echo "ERROR: {{_ctr}} not running on thor. Run: just up"; exit 1; }'

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
compile: _preflight
    @ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "mkdir -p {{_bdir}} && cd {{_bdir}} && ([ -f CMakeCache.txt ] || cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_ONLY_EXTENSION=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache {{_wdir}}) && cmake --build . && cmake --install .") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

# Print ccache statistics from the dev container.
ccache-stats:
    @ssh {{_thor}} 'docker exec {{_ctr}} ccache --show-stats | head -20'

# Fast static analysis: clang-format dry-run over src/ (parallel-safe).
check: _preflight
    @ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_check_q}}.sock && id=$(tsp docker exec {{_ctr}} bash -c "cd {{_wdir}} && make fcheck") && echo "[job $id queued on {{_check_q}}]" && tsp -c $id'

# Run PostgreSQL regression tests (queued), gated against
# tests/pg_regress_baseline.txt. Exits non-zero only on *delta* vs the
# baseline (new failures, or previously-failing tests that now pass).
# ctest runs as the postgres user so pg_regress's default "whoami"
# connection works.
test: _preflight
    #!/usr/bin/env bash
    set -euo pipefail
    ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "mkdir -p /tmp/pgx_ir && chmod 777 /tmp/pgx_ir; chmod o+x /workspace/.worktrees 2>/dev/null || true; chmod -R o+rX {{_wdir}}; chown -R postgres:postgres {{_bdir}} && cd {{_bdir}} && (su postgres -c \"ctest -V\" 2>&1 | tee /tmp/ctest.out; cat /tmp/ctest.out | python3 {{_wdir}}/scripts/ptest_with_baseline.py --baseline-file {{_wdir}}/tests/pg_regress_baseline.txt)") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

# Re-record the pg_regress baseline. Run only when you have consciously
# accepted a new set of red tests on main — each entry that gets added
# here must be justified in the PR body. Removing entries is free (those
# tests are now passing).
test-record-baseline: _preflight
    #!/usr/bin/env bash
    set -euo pipefail
    ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "mkdir -p /tmp/pgx_ir && chmod 777 /tmp/pgx_ir; chmod o+x /workspace/.worktrees 2>/dev/null || true; chmod -R o+rX {{_wdir}}; chown -R postgres:postgres {{_bdir}} && cd {{_bdir}} && (su postgres -c \"ctest -V\" 2>&1 | tee /tmp/ctest.out; cat /tmp/ctest.out | python3 {{_wdir}}/scripts/ptest_with_baseline.py --baseline-file {{_wdir}}/tests/pg_regress_baseline.txt --record)") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

# Smoke benchmark: SF=0.01, 5 iterations per query, pgx ON vs OFF. ~40s total.
# iter=5 damps JIT-compile variance which is the dominant noise source at
# small SF — the chart would be useless with iter=1 (±8% run-to-run).
bench: _preflight
    @ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "cd {{_wdir}} && python3 benchmark/tpch/run.py 0.01 --port 5432 --container {{_ctr}} --indexes --skip q17,q20 --iterations 5") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

# Trustworthy-signal benchmark: SF=0.16, 3 iterations. ~60-120s. Run before
# merging anything that claims a performance improvement — SF=0.01 is
# compile-dominated and can't distinguish real speedups from noise.
bench-merge: _preflight
    @ssh {{_thor}} 'export TS_SOCKET=/tmp/{{_build_q}}.sock && tsp -S 1 >/dev/null && id=$(tsp docker exec {{_ctr}} bash -c "cd {{_wdir}} && python3 benchmark/tpch/run.py 0.16 --port 5432 --container {{_ctr}} --indexes --skip q17,q20 --iterations 3") && echo "[job $id queued on {{_build_q}}]" && tsp -c $id'

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
    # Fetch the baseline from origin/main without polluting the feature branch.
    git fetch origin main --quiet
    baseline_path=$(git ls-tree -r --name-only origin/main -- 'benchmarks/pr-*.db' | sort -V | tail -1 || true)
    if [ -z "${baseline_path}" ]; then
        echo "ERROR: no baseline on origin/main:benchmarks/. Merge at least one PR first." >&2
        exit 1
    fi
    baseline_name=$(basename "${baseline_path}")
    # Stage the baseline on thor (not committed here — just in /tmp for report.py).
    git show "origin/main:${baseline_path}" > /tmp/${baseline_name}
    scp -q /tmp/${baseline_name} {{_thor}}:/tmp/${baseline_name}
    rm /tmp/${baseline_name}
    ssh {{_thor}} "docker cp /tmp/${baseline_name} {{_ctr}}:/tmp/${baseline_name}"
    ssh {{_thor}} "docker exec {{_ctr}} python3 {{_wdir}}/benchmark/report.py \
        --baseline /tmp/${baseline_name} \
        --current {{_wdir}}/benchmarks/${slug}.db \
        --out {{_wdir}}/benchmarks/${slug} \
        --chart-url \"https://raw.githubusercontent.com/zyros-dev/pgx-lower/${branch}/benchmarks/${slug}.png\""
    echo ""
    echo "Artifacts: benchmarks/${slug}.{db,png,md}"
    echo "Baseline : ${baseline_name} (from origin/main)"

# --- Queue ops ------------------------------------------------------------

# Show the build queue (pending + running + recent finished jobs).
queue:
    @echo "=== build queue ==="
    @ssh {{_thor}} 'TS_SOCKET=/tmp/{{_build_q}}.sock tsp' || true
    @echo "=== check queue ==="
    @ssh {{_thor}} 'TS_SOCKET=/tmp/{{_check_q}}.sock tsp' || true

# Tail the live output of a running job from the build queue.
tail ID:
    ssh {{_thor}} 'TS_SOCKET=/tmp/{{_build_q}}.sock tsp -t {{ID}}'

# Cancel and remove a queued or running build job.
cancel ID:
    ssh {{_thor}} 'TS_SOCKET=/tmp/{{_build_q}}.sock tsp -k {{ID}} || true; TS_SOCKET=/tmp/{{_build_q}}.sock tsp -r {{ID}}'

# --- Worktree lifecycle ----------------------------------------------------

# Create a git worktree on mac + thor and a mutagen sync session between them.
# Usage: just worktree-new feat-foo
worktree-new NAME:
    @test -n "{{NAME}}" || { echo "NAME required"; exit 1; }
    git worktree add .worktrees/{{NAME}}
    ssh {{_thor}} 'cd ~/repos/pgx-lower && git fetch origin && git worktree add .worktrees/{{NAME}} 2>/dev/null || true'
    mutagen sync create \
        --name=pgx-lower-{{NAME}} \
        --sync-mode=two-way-resolved \
        --ignore='/build-*/' --ignore='/build-docker-*/' --ignore='/postgres-debug/' \
        --ignore='__pycache__/' --ignore='*.pyc' --ignore='*.tar.gz' \
        --ignore='/.venv/' --ignore='/.idea/' --ignore='/.vscode/' \
        .worktrees/{{NAME}} {{_thor}}:/home/zel/repos/pgx-lower/.worktrees/{{NAME}}
    @echo ""
    @echo "Worktree ready. On mac: cd .worktrees/{{NAME}}"
    @echo "Mutagen session: pgx-lower-{{NAME}}"

# Tear down a worktree and its sync.
worktree-rm NAME:
    -mutagen sync terminate pgx-lower-{{NAME}}
    -ssh {{_thor}} 'cd ~/repos/pgx-lower && git worktree remove --force .worktrees/{{NAME}}'
    -git worktree remove --force .worktrees/{{NAME}}

# List worktrees + their sync sessions.
worktree-list:
    @echo "=== mac worktrees ==="
    @git worktree list
    @echo ""
    @echo "=== thor worktrees ==="
    @ssh {{_thor}} 'cd ~/repos/pgx-lower && git worktree list'
    @echo ""
    @echo "=== mutagen sessions ==="
    @mutagen sync list | grep -E '^Name:|^Status:' || true

# --- PR --------------------------------------------------------------------

# Open a PR from the current branch against main. Fill in the Summary +
# stats summary before requesting review.
pr TITLE:
    gh pr create --base main --head "$(git rev-parse --abbrev-ref HEAD)" --title "{{TITLE}}" --body "$(printf '## Summary\n\n<what and why>\n\n## Stats summary\n\n<paste the stats summary block here — required>\n\n## Test plan\n- [ ] just check\n- [ ] just test\n- [ ] just bench\n')"

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
spec-claim NN BRANCH OWNER='':
    OWNER="{{OWNER}}" python3 scripts/spec_status.py claim "{{NN}}" "{{BRANCH}}"

# Release a claim (e.g. abandoning the work). Sets state back to available.
spec-release NN:
    python3 scripts/spec_status.py release "{{NN}}"

# Mark a spec done after its PR merges. PR is the PR number.
spec-complete NN PR:
    python3 scripts/spec_status.py complete "{{NN}}" "{{PR}}"

# Mark a spec as in_review (PR open). PR is the PR number.
spec-in-review NN PR:
    python3 scripts/spec_status.py in_review "{{NN}}" "{{PR}}"

# Mark a spec as blocked, with a reason note.
spec-block NN REASON:
    python3 scripts/spec_status.py block "{{NN}}" "{{REASON}}"
