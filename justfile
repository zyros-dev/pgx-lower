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
# Emits a one-line "BUILD OK" / "BUILD FAILED" verdict at the end so agents
# don't have to infer success by reading the full cmake/ninja scroll.
compile: _preflight
    #!/usr/bin/env bash
    set -o pipefail
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
    #!/usr/bin/env bash
    set -euo pipefail
    test -n "{{NAME}}" || { echo "NAME required"; exit 1; }
    # Refuse to silently reuse an existing local branch. If it exists, the
    # previous work on it is still there (even after worktree-rm); `git
    # worktree add` without -b would quietly check that out, and an agent
    # expecting a fresh branch would get the abandoned implementation back.
    # Best recovery: `just spec-abandon` for specs (nukes local branch too),
    # or `git branch -D {{NAME}}` if you really want to reuse the name.
    if git show-ref --verify --quiet "refs/heads/{{NAME}}"; then
        echo "ERROR: local branch '{{NAME}}' already exists." >&2
        echo "  For specs: run 'just spec-abandon NN \"<reason>\"' to clean up atomically." >&2
        echo "  Otherwise: 'git branch -D {{NAME}}' to discard it, then retry." >&2
        exit 1
    fi
    git fetch origin main --quiet
    git worktree add -b "{{NAME}}" ".worktrees/{{NAME}}" origin/main
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

# Mark a spec done after its PR merges. PR is the PR number.
spec-complete NN PR:
    python3 scripts/spec_status.py complete "{{NN}}" "{{PR}}"

# Mark a spec as in_review (PR open). PR is the PR number.
spec-in-review NN PR:
    python3 scripts/spec_status.py in_review "{{NN}}" "{{PR}}"

# Mark a spec as blocked, with a reason note.
spec-block NN REASON:
    python3 scripts/spec_status.py block "{{NN}}" "{{REASON}}"
