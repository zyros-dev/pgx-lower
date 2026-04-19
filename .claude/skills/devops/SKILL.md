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
just bench
```

SF=0.01 TPC-H A/B run, pgx ON vs OFF, under ~30s. Read the per-query table and the geomean. If a query you touched regressed, note it — don't silently ignore.

Skipped queries: q17 and q20 are skipped by default (pathologically slow without indexes at small SF). If your change touches code paths those queries exercise, run the `full` profile separately and call it out in the report.

## 6. Final report

Before opening the PR, produce a short write-up covering:

- **What changed** (files, high-level)
- **Why** (the test you added + the observation that motivated it)
- **Results**:
  - `just check` — clean / hits
  - `just test` — pass count / any new failures
  - `just bench` — per-query deltas for queries you touched, plus the geomean
- **Stats summary** — required block (separate spec describes the format; if
  the format isn't documented yet, ask the user). Goes into the PR body's
  "Stats summary" section.
- **Follow-ups** — anything you noticed but didn't fix

## 7. Pull request

```
git add -A
git commit -m "$ARGUMENTS: <one-line>"
git push -u origin $ARGUMENTS
just pr "<title>"
```

`just pr` uses a templated body (Summary + Stats summary + Test plan checklist). Paste the final report from step 6 into the Summary and Stats summary sections before requesting review.

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
| `just bench` shows a regression on queries you touched | Don't silently ignore. Note in report and ask user before opening PR. |
| Mutagen sync conflict | `mutagen sync flush pgx-lower-<slug>`; resolve manually; commit. |
| Spec genuinely ambiguous | Ask user; if they don't know either, run `just spec-block NN "<reason>"` and stop. |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: task-spooler not installed` | tsp missing on thor | `just bootstrap-tsp` (one-time, needs sudo) |
| `ERROR: pgx-lower-dev not running` | container stopped | `just up` |
| `compile` queues but never runs | tsp daemon stuck | `ssh comfy 'TS_SOCKET=/tmp/pgx-build.sock tsp -K'` to kill daemon; it respawns on next use |
| Test passes that shouldn't | stale `.so` in PG | `just compile` re-installs the extension into the container's PG; rerun `just test` |
| Mutagen shows conflicts | simultaneous edit mac/thor | `mutagen sync flush pgx-lower-<slug>`; resolve manually, commit |
