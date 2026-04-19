---
name: devops
description: End-to-end playbook for implementing a feature or fix in pgx-lower. Auto-invoke when the user says "start spec NN", "implement spec NN", or asks for any code change that should end in a pull request. Covers spec claiming, worktree creation, TDD, build, static analysis, tests, benchmark, PR, and status-board upkeep. The user only ever says "start spec NN" — every other step is yours.
disable-model-invocation: false
argument-hint: "<spec number, e.g. 03> OR <short feature slug, e.g. feat-trim-proj>"
allowed-tools: "Bash(just *) Bash(git *) Bash(gh *) Bash(ssh comfy *) Bash(mutagen *) Bash(cd *)"
---

# devops — implement-and-ship playbook

The whole pipeline is wrapped in `justfile` recipes that SSH into thor and funnel through the `tsp` queue so concurrent agents don't OOM the box. You almost never call `ssh` or `docker exec` directly — use the recipes.

**Everything runs on thor.** See CLAUDE.md for the sync/toolchain invariants.

**The user does not orchestrate.** They say "start spec NN" and you do every step below, including the status-board updates. Don't ask the user for the branch name, the slug, the worktree, or whether to claim — you decide and do it. Only escalate when something blocks (claim conflict, spec genuinely ambiguous, build broken in a way you can't diagnose).

## 0. Preflight (always)

Run these. They're fast.

```
just queue          # build-queue depth
just worktree-list  # other agents' worktrees
just spec-status    # the spec board
```

Concrete queue-depth thresholds. `just queue` emits `N total (M running, K queued)` as its summary line — use the total to decide:

| Total jobs | What to expect |
|---|---|
| 0–2 | Fine. Your job runs ~immediately. |
| 3–5 | Noticeable wait (minutes per slot). Plan a bigger change per build; avoid chaining `compile && utest && test` submissions in tight succession. |
| 6+  | Deep queue. Run `just check-diff` locally while the queue drains — it lives on a separate queue (`pgx-check`, not `pgx-build`) and runs in parallel, so you get format feedback without joining the backlog. |

Don't bypass the queue. If a worktree for the spec already exists with a different owner, stop and tell the user.

## 1. Claim the spec + create the worktree

When the user says "start spec NN":

1. Read `specs/NN-*.md` so you know what you're doing before you reserve a slot.
2. Pick the branch slug. Convention: `spec-NN-<short-keyword>` derived from the spec filename (e.g. `spec-03-cache`, `spec-05-decode`). Lowercase, hyphens.
3. From the **main** checkout (not a worktree), claim and create the worktree atomically:

```
cd ~/repos/pgx-lower                        # main checkout — required for spec-claim
just spec-claim NN <slug>                    # rebases main, marks spec in_progress, pushes
just worktree-new <slug>                     # creates worktree AND the <slug> branch on it
cd ~/repos/pgx-lower/.worktrees/<slug>       # always absolute; every `just` recipe reads
                                             # invocation_directory() freshly, so a relative
                                             # `cd .worktrees/<slug>` can fail or get you into
                                             # the wrong tree depending on cwd at call time.
```

**Do not run `git checkout -b <slug>` after `worktree-new`** — the recipe already created and checked out the branch. `git checkout -b` would fail with "branch already exists" and trying to debug around it is a waste. Skip straight to step 2.

If `just spec-claim` fails because the spec is already claimed, first check whether the claim is stale:

- `gh pr view <PR-from-status> --json state,title` — PR OPEN, CLOSED, or MERGED?
- `gh pr list --head <branch-from-status>` — any activity?
- `mutagen sync list pgx-lower-<branch>` — session still live?

`just spec-abandon NN "<reason>"` is the right tool for *every* case where you need to reset state — not just zombies. It closes the PR (open or closed), retitles it `[abandoned] spec NN — <reason>`, deletes the remote branch, tears down the worktree on mac + thor, terminates the mutagen session, and flips STATUS back to `available`. Fully idempotent.

Use it when:
- PR is CLOSED unmerged + worktree dangling ("zombie state").
- PR is OPEN but the user told you to restart / redo the spec.
- Local branch exists from a previous attempt and `just worktree-new` is failing with "branch already exists."

Do NOT use it when someone else is actively working on the spec and you don't have permission — that's where you stop and ask the user. Everything else is a `spec-abandon` away from a clean re-claim.

## 2. Red — write a failing test first

TDD is mandatory here (see CLAUDE.md). Before any implementation change:

1. **Pick the right test home — default to `tests/unit/` (gtest), not `tests/sql/`.**

   See **`tests/unit/README.md`** for the decision table and existing-test patterns. Short version:

   - `tests/sql/` + `tests/expected/` is the **pg_regress output-equivalence suite** — curated coverage meant to stay stable. Adding a new `.sql` per spec is bloat; the existing suite already catches correctness regressions for lowering changes.
   - `tests/unit/test_lowerings/*.cpp` (gtest) is the **primary TDD home** for MLIR passes, dialect patterns, JIT pipeline, and pure-computation utilities (type mapping, cost formulas, plan-shape hashing). There are three existing smoke tests you can copy as a template: `test_pipeline_phases.cpp` (full pipeline), `test_boolean_lowering.cpp` (single pattern), `test_type_mapping.cpp` (pure computation).
   - **Runtime FFI code** (`src/pgx-lower/runtime/*`, `tuple_access`, `PostgreSQLRuntime`) is integration-bound — it IS the PG boundary. Unit tests there would be parallel-reimplementations of PG. Those stay as pg_regress tests; don't try to unit-test them.

   **Add to `tests/sql/` only if**: your change introduces a SQL-level feature the existing suite genuinely doesn't cover. Default assumption: you're adding a unit test under `test_lowerings/`.

2. Add or modify the test that captures the new behavior.

   - **Unit tests** (preferred): write the `.cpp` in `tests/unit/test_lowerings/`, add to `tests/unit/test_lowerings/CMakeLists.txt`. `just test` runs the full suite.
   - **pg_regress** (only when actually needed): write the `.sql` file, but **do NOT write the `.out` file by hand**. pg_regress's output format has footguns (SQL lines get echoed, `IF NOT EXISTS` emits NOTICE messages, column headers often have trailing whitespace that editors strip).

     Canonical pg_regress authoring flow:

     ```
     # 1. Add tests/sql/NN_name.sql (no .out file yet).
     # 2. Run the tests to generate the actual pg_regress output.
     just test                              # will fail: no expected/NN_name.out
     # 3. Copy pg_regress's own output back as the authoritative expected file.
     just expected-from-results NN_name
     # 4. Re-run to confirm green.
     just test
     ```

     `just expected-from-results NN_name` reads `build-docker-ptest/extension/results/NN_name.out` on thor and writes `tests/expected/NN_name.out` verbatim. Always use it for the first commit of a new regression test — the manual-authoring path burns time on whitespace and NOTICE-line debugging that pg_regress already knows how to produce.

3. Confirm it **fails** for the reason you expect:
   - Unit tests: `just utest` (fast; configures+builds to `build-docker-utest/` on first run, incremental after).
   - pg_regress: `just test` (slower; gated against `tests/pg_regress_baseline.txt`).

   If the test passes, it's not covering what you think — fix the test before touching code.

## 3. Green — minimum change to pass

Implement. Keep the change scoped to what the failing test demands; resist refactoring until green.

Iterate tightly:

```
just compile    # ~seconds when cached; streams compiler errors live
just utest      # fast TDD loop — gtest-based, scoped to the thing you changed
just test       # pg_regress output-equivalence; run before opening the PR
```

`compile`, `utest`, and `test` all share the serialized build queue, so they don't stomp on each other or a concurrent bench. Iterate `just compile && just utest` tightly while implementing; run `just test` once before opening the PR to confirm no pg_regress regression.

**`just compile` is NOT a RED signal for extension-boundary symbols.** When a SQL function is declared as `AS '$libdir/pgx_lower', 'symbol_name'`, the C symbol is looked up via `dlsym` at PG `CREATE FUNCTION` / invocation time — no other translation unit references it at compile time, so the link resolves whether the symbol exists or not. A `BUILD OK` from `just compile` does NOT prove the implementation is present. The RED signal for this class of change is `just test` reporting `ERROR: function X() does not exist` or `could not find function "X" in file ".../pgx_lower.so"`. Treat those as the canonical failing-test signal — and when you wire a new `CREATE FUNCTION ... AS '$libdir/pgx_lower', 'foo'`, don't declare the implementation done until `just test` is green on a case that actually calls `foo()`.

**Mutagen flush is automatic — don't `sleep`.** Every recipe that hits thor now runs through `_preflight`, which calls `mutagen sync flush "pgx-lower-<branch>"` before the build/test/bench command executes. The flush is ~100ms when the session is idle and deterministic — it blocks until the mac→thor cycle is complete, so `ninja` / `pg_regress` / `run.py` see the file mtimes you just edited locally. **Skip the old `sleep 3` cargo-cult pattern**: editing a file on mac and running `just compile` back-to-back is safe. The only time you need an explicit flush is when you're bypassing the recipes (e.g. calling `ssh comfy docker exec …` directly), which you shouldn't be doing anyway.

## 4. Static analysis

```
just check-diff      # PR gate: clang-format only on lines your PR changed
just ffix-diff       # auto-fix those same lines in place
just check           # whole-tree; for reference only (noisy)
```

**Use `just check-diff` for the PR gate.** It runs `clang-format-diff-20` against the unified diff vs `origin/main` — output is scoped to the exact lines your PR added or modified, *not* the whole file. That means touching a file with pre-existing formatting debt (e.g. `executor_c.c`) doesn't flood the output with violations you're not responsible for.

- **Clean** → `check-diff: clean (your hunks match the project style)` → you're good.
- **Hits** → the recipe prints the suggested reformat as a diff. Apply with `just ffix-diff`, which does the same in-place. Re-run `check-diff` to confirm clean.

`just check` (whole-tree) is only useful as a historical reference; don't gate on it — it's not achievable repo-wide right now.

## 5. Benchmark

```
just bench           # SF=0.5, iter=1. ~5 min cold, ~3 min cached. Default for every PR.
just bench-report    # snapshot + chart + verdict + auto-inject into PR body
```

`just bench` runs TPC-H at **SF=0.5, 1 iteration**. SF=0.5 puts us in the execution-dominated regime (~88% of each query's wall time is real work, ~12% is JIT compile), which drops per-query variance from the ~50% bimodal chaos we saw at SF=0.01 to honest 5–10% execution noise. Medianing iterations doesn't meaningfully reduce execution-scale variance; iter=1 is correct at this SF.

**Idempotent caching.** `run.py` auto-detects whether TPC-H is already loaded at the target SF (via the `customer` row count — exactly `150_000 * SF`, no jitter) and skips the ~3-minute dbgen + psql-load step when it is. First `just bench` on a fresh container takes ~5 min; subsequent runs land in ~3 min. No flag, no sentinel — the check runs every time. Cache state lives in the `postgres-data` docker volume so it persists across container restarts.

**When to also run `just bench-merge`.** Before merging anything claiming a perf improvement where SF=0.5 numbers feel marginal. It runs SF=1 iter=1 (~10 min cold, ~6 min cached) — per-query wall times are seconds, so ±5% is real signal.

**Skipped queries.** q17 and q20 are skipped by default — pathologically slow without indexes. If your change touches code paths they exercise, run the `full` profile in `benchmark-config.yaml` separately and call it out.

## 6. Benchmark report + verdict

`just bench-report` consumes `benchmark/output/benchmark.db` from the last `just bench`, names it using the PR number, and emits:

- `benchmarks/pr-<N>-spec-<NN>-<slug>.db` — raw data (one per PR, committed to the feature branch only).
- `benchmarks/pr-<N>-spec-<NN>-<slug>.png` — bar chart: one bar per TPC-H query, up = PR faster than baseline (green), down = regression (red), gray inside ±1% noise.
- `benchmarks/pr-<N>-spec-<NN>-<slug>.md` — verdict header + summary + per-query table with **PG reference column**.

Naming rules:
- Spec branch `spec-NN-<slug>` → `pr-<N>-spec-<NN>-<slug>.db`
- Non-spec branch → `pr-<N>-<slug>.db`

**Ordering matters:** `just bench-report` needs an open PR to know its PR number. Call order is `just bench → just pr → just bench-report`. The last step auto-injects the bench block into the PR body — you do not re-paste it manually.

The **baseline** db is pulled on the fly from `origin/main:benchmarks/` (the alphanumerically latest `pr-*.db` there, i.e. the most recently merged PR's db). Feature branches never commit the baseline — each PR adds exactly one `.db` (its own). When this PR merges, its db lands on main and becomes the baseline for future PRs.

**Verdict lines are auto-computed** and go at the top of the `.md`:

| Verdict | Rule |
|---|---|
| 🟢 YAY    | geomean ≥ +3% AND no query regresses worse than −5% |
| 🔴 NAY    | geomean ≤ −3% OR any query regresses worse than −10% |
| 🟡 MAYBE  | everything in between (inside noise band, or mixed) |

If the verdict is NAY, **stop and tell the user before marking the PR ready.** Don't paper over regressions. If MAYBE, it's your call — if the change is correctness-motivated (bench is just a regression gate), proceed; if it's supposed to be a perf win, run `just bench-merge` for trustworthy signal first.

## 7. Pull request

Because `just bench-report` needs the PR number to name its artifacts, the PR is opened *first*, then the bench artifacts are committed in a follow-up commit on the same branch:

```
git add -A                           # code + tests only at this point
git commit -m "$ARGUMENTS: <one-line>"
git push -u origin $ARGUMENTS
just pr "<title>"                    # PR opens — now you know N
just bench-report                    # emits benchmarks/pr-N-spec-NN-<slug>.{db,png,md}
git add benchmarks/pr-${N}-*
git commit -m "Benchmark report for PR #${N}"
git push
```

`just bench-report` **auto-injects the `.md` block into the PR body**, replacing the `<paste the stats summary block here — required>` placeholder that `just pr` left. You don't have to `gh pr edit` it in by hand. What you still do manually:

- Fill in the `<what and why>` Summary placeholder — `bench-report` can't know your intent.
- Keep the `## Test plan` checklist from the `just pr` template and tick items as they pass.

If you re-run `bench-report` (e.g. after `just bench-merge`), it detects the placeholder is already gone and skips the auto-inject — paste the new `.md` manually or edit the PR body to match.

If you ran `just bench-merge` too, run `just bench-report` again afterwards — it overwrites the `.db/.png/.md` triplet with the higher-SF numbers, so the PR body picks up the authoritative verdict.

Capture the PR number from `gh pr view --json number -q .number` or from the URL `gh pr create` printed.

Then mark the spec as `in_review` — from the **main** checkout:

```
cd ~/repos/pgx-lower
just spec-in-review NN <pr-number>
```

Tell the user the PR is up and ready for review (paste the URL).

## 8. After merge (in any future session)

When the user says "spec NN merged" or you observe via `gh pr view NN --json state -q .state` that the PR is `MERGED`:

```
cd ~/repos/pgx-lower
just spec-complete NN <pr-number>
just worktree-rm <slug>
```

`worktree-rm` terminates the mutagen session and removes the worktree on both mac and thor. `spec-complete` flips the board row to `done` so future agents see the dependency clear.

## When things go wrong

| Situation | What you do |
|-----------|-------------|
| `just spec-claim` says spec is already claimed (CLOSED PR, OPEN PR, stale local branch — any combination) | `just spec-abandon NN "<reason>"` cleans EVERY case atomically, then re-claim. The recipe works on live PRs too (closes + retitles them). Only stop-and-ask if someone else is actively working on it. |
| `just compile` fails with errors you can fix | Fix and re-run. |
| `just compile` fails with errors you can't diagnose | Tell user; include the last 30 lines of output. |
| `just test` shows a failure that's the test you wrote (Red phase) | Continue to Green. |
| `just test` shows unrelated failures | Don't proceed. Tell user; don't paper over. |
| `just bench-report` emits 🔴 NAY | Real regression (geomean down ≥3% OR any query past −10%, OR correctness failure). At SF=0.5 this is execution-scale signal, not compile noise — treat as real. Stop; tell the user. Do not mark the PR ready until understood. |
| `just bench-report` emits 🟡 MAYBE on a perf-claiming change | Run `just bench-merge` (SF=1) for a trustworthy signal before marking the PR ready. |
| `just bench-report` emits 🟡 BASELINE-SF-MISMATCH — unable to compare | Baseline and current were captured at different scale factors; per-query percentages are meaningless. Happens automatically when the canonical bench SF changes on main. Not an error; note it in the PR body and proceed. Subsequent PRs will have a same-SF baseline. |
| `just bench-report` says "no baseline on origin/main" | **First-PR bootstrap case.** The recipe auto-falls-back to a self-compare — your PR will show 0% deltas but validates the full pipeline and seeds the baseline for future PRs. Not an error; note it in the PR body and proceed. |
| Mutagen sync conflict | `mutagen sync flush pgx-lower-<slug>`; resolve manually; commit. |
| Spec genuinely ambiguous | Ask user; if they don't know either, run `just spec-block NN "<reason>"` and stop. |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: task-spooler not installed` | tsp missing on thor | `just bootstrap-tsp` (one-time, needs sudo) |
| `ERROR: pgx-lower-dev not running` | container stopped | `just up` |
| `compile` queues but never runs | tsp daemon stuck | `ssh comfy 'TS_SOCKET=/tmp/pgx-build.sock tsp -K'` to kill daemon; it respawns on next use |
| `Job N not finished or not running` from tsp under queue contention | tsp state confusion when another agent's slot holder exits weirdly | Just re-run the recipe; tsp is idempotent and the second submission gets a fresh id. |
| Test passes that shouldn't | stale `.so` in PG | `just compile` re-installs the extension into the container's PG; rerun `just test` |
| Mutagen shows conflicts | simultaneous edit mac/thor | `mutagen sync flush pgx-lower-<slug>`; resolve manually, commit |
| `ninja: no work to do` after editing source | rare mtime race; normally `_preflight` flushes mutagen before ninja runs, but a file touched mid-recipe can slip through | `ssh comfy 'docker exec pgx-lower-dev touch /workspace/.worktrees/<slug>/<path/to/file>'` to nudge ninja. Don't `sleep` — the flush is already automatic. |
| `just check` emits thousands of violations unrelated to your diff | Pre-existing clang-format debt; whole tree hasn't been formatted | Gate on "no *new* violations on files your diff touches". `just check` clean is not yet achievable repo-wide. |

## Gotchas worth internalizing

- **Bash cwd is not sticky across tool calls.** Every `just` recipe reads `invocation_directory()` which is the shell's cwd *at the moment of the call*. If you don't explicitly `cd .worktrees/<slug>` (or pass the cwd via your tool's working-directory option) each time, `just` runs against the main repo — which silently succeeds and builds/tests the wrong tree. Double-check the resolved path with `just --dry-run compile | head -1` if something's off.
- **First bench-report self-compares.** When `origin/main:benchmarks/` has no `pr-*.db` yet, `bench-report` falls back to self-comparison (0% deltas, MAYBE verdict) so the artifacts still land and seed the baseline. Don't treat it as a failure.
