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

If `just queue` shows a deep queue, expect your jobs to wait — don't bypass.
If a worktree for the spec already exists with a different owner, stop and tell the user.

## 1. Claim the spec + create the worktree

When the user says "start spec NN":

1. Read `specs/NN-*.md` so you know what you're doing before you reserve a slot.
2. Pick the branch slug. Convention: `spec-NN-<short-keyword>` derived from the spec filename (e.g. `spec-03-cache`, `spec-05-decode`). Lowercase, hyphens.
3. From the **main** checkout (not a worktree), claim and create the worktree atomically:

```
cd ~/repos/pgx-lower      # main checkout — required for spec-claim
just spec-claim NN <slug>  # rebases main, marks spec in_progress, pushes
just worktree-new <slug>
cd .worktrees/<slug>
git checkout -b <slug>
```

If `just spec-claim` fails because the spec is already claimed, tell the user. Don't double-claim.

## 2. Red — write a failing test first

TDD is mandatory here (see CLAUDE.md). Before any implementation change:

1. Identify the test harness for the thing you're changing — PostgreSQL regression tests live in `extension/sql/` + `extension/expected/`, unit tests in `tests/`.
2. Add or modify a test that captures the new behavior.
3. `just test` — confirm it **fails** for the reason you expect. If it passes, the test isn't covering what you think.

## 2. Red — write a failing test first

TDD is mandatory here (see CLAUDE.md). Before any implementation change:

1. Identify the test harness for the thing you're changing — PostgreSQL regression tests live in `extension/sql/` + `extension/expected/`, unit tests in `tests/`.
2. Add or modify a test that captures the new behavior.
3. `just test` — confirm it **fails** for the reason you expect. If it passes, the test isn't covering what you think.

## 3. Green — minimum change to pass

Implement. Keep the change scoped to what the failing test demands; resist refactoring until green.

Iterate tightly:

```
just compile    # ~seconds when cached; streams compiler errors live
just test       # runs regression tests once compile succeeds
```

`just compile` and `just test` share a serialized queue — they don't stomp on each other or on a concurrent bench. If a test fails, read the output, fix, repeat. If compile fails, fix and re-run; cached object files survive.

## 4. Static analysis

```
just check
```

Fast clang-format dry-run. Runs on a separate queue so it doesn't block builds. Fix any hits with `ffix` inside the container if needed (open `just` for the full list).

## 5. Benchmark

```
just bench           # SF=0.01, iter=5, ~40s. Default for every PR.
just bench-report    # snapshot + chart + verdict
```

`just bench` runs 5 iterations per query per mode because at SF=0.01 pgx-lower's JIT compile dominates execution (q02: PG 8ms vs pgx 570ms — essentially all compile). One iteration has ~±8% run-to-run variance; iter=5's median gets that down to ~±3%, which is what the verdict thresholds below are calibrated against.

**When to also run `just bench-merge`.** Before merging anything that claims a performance improvement. It runs SF=0.16 with iter=3 (~60-120s) so query execution starts mattering and the numbers aren't compile-noise. Paste both chart links into the PR body if you run it.

**Skipped queries.** q17 and q20 are skipped by default — pathologically slow without indexes at small SF. If your change touches code paths they exercise, run the `full` profile in `benchmark-config.yaml` separately and call it out.

## 6. Benchmark report + verdict

`just bench-report` consumes `benchmark/output/benchmark.db` from the last `just bench`, names it using the PR number, and emits:

- `benchmarks/pr-<N>-spec-<NN>-<slug>.db` — raw data (one per PR, committed to the feature branch only).
- `benchmarks/pr-<N>-spec-<NN>-<slug>.png` — bar chart: one bar per TPC-H query, up = PR faster than baseline (green), down = regression (red), gray inside ±1% noise.
- `benchmarks/pr-<N>-spec-<NN>-<slug>.md` — verdict header + summary + per-query table with **PG reference column**.

Naming rules:
- Spec branch `spec-NN-<slug>` → `pr-<N>-spec-<NN>-<slug>.db`
- Non-spec branch → `pr-<N>-<slug>.db`

**Ordering matters:** `just bench-report` needs an open PR to know its PR number. Call order is `just bench → just pr → just bench-report → paste the .md into the PR body`.

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

Then edit the PR body to paste in the contents of `benchmarks/pr-<N>-spec-<NN>-<slug>.md` — that supplies the chart (via raw.githubusercontent URL), verdict, and per-query table. Add a **Summary** section above it (what changed, why), and keep the `## Test plan` checklist from the `just pr` template.

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
| `just spec-claim` says spec is already claimed | Stop, tell user; don't override. |
| `just compile` fails with errors you can fix | Fix and re-run. |
| `just compile` fails with errors you can't diagnose | Tell user; include the last 30 lines of output. |
| `just test` shows a failure that's the test you wrote (Red phase) | Continue to Green. |
| `just test` shows unrelated failures | Don't proceed. Tell user; don't paper over. |
| `just bench-report` emits 🔴 NAY | Stop; tell the user. Do not open the PR until the regression is understood. |
| `just bench-report` emits 🟡 MAYBE on a perf-claiming change | Run `just bench-merge` for a trustworthy signal before opening the PR. |
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
| `ninja: no work to do` after editing source | mutagen mtime race — file synced with older mtime than its build product | `ssh comfy 'docker exec pgx-lower-dev touch /workspace/.worktrees/<slug>/<path/to/file>'` to nudge ninja; or wait ~2s before running `just compile` |
| `just check` emits thousands of violations unrelated to your diff | Pre-existing clang-format debt; whole tree hasn't been formatted | Gate on "no *new* violations on files your diff touches". `just check` clean is not yet achievable repo-wide. |

## Gotchas worth internalizing

- **Bash cwd is not sticky across tool calls.** Every `just` recipe reads `invocation_directory()` which is the shell's cwd *at the moment of the call*. If you don't explicitly `cd .worktrees/<slug>` (or pass the cwd via your tool's working-directory option) each time, `just` runs against the main repo — which silently succeeds and builds/tests the wrong tree. Double-check the resolved path with `just --dry-run compile | head -1` if something's off.
- **First bench-report self-compares.** When `origin/main:benchmarks/` has no `pr-*.db` yet, `bench-report` falls back to self-comparison (0% deltas, MAYBE verdict) so the artifacts still land and seed the baseline. Don't treat it as a failure.
