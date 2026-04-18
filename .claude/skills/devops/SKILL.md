---
name: devops
description: End-to-end playbook for implementing a feature or fix in pgx-lower. Use whenever the user asks you to make a code change that should end in a pull request. Covers worktree creation, TDD, build, static analysis, tests, benchmark, and PR.
disable-model-invocation: false
argument-hint: "<short feature slug, e.g. feat-trim-proj>"
allowed-tools: "Bash(just *) Bash(git *) Bash(gh *) Bash(ssh comfy *) Bash(mutagen *)"
---

# devops — implement-and-ship playbook

The whole pipeline is wrapped in `justfile` recipes that SSH into thor and funnel through the `tsp` queue so concurrent agents don't OOM the box. You almost never call `ssh` or `docker exec` directly — use the recipes.

**Everything runs on thor.** See CLAUDE.md for the sync/toolchain invariants.

## 0. Before you start

- `just queue` — glance at the build queue. If it's deep, expect your `compile`/`test`/`bench` to wait; don't bypass.
- `just worktree-list` — confirm you're not about to collide with another agent's worktree.

## 1. Create a worktree

Never work on `main` directly. Pick a short slug (lowercase, hyphens), create an isolated worktree with its own mutagen sync:

```
just worktree-new $ARGUMENTS
cd .worktrees/$ARGUMENTS
git checkout -b $ARGUMENTS
```

This creates `.worktrees/<slug>/` on both mac and thor, and a dedicated mutagen session `pgx-lower-<slug>` so your edits sync without touching the main repo.

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
- **Follow-ups** — anything you noticed but didn't fix

## 7. Pull request

```
git add -A
git commit -m "$ARGUMENTS: <one-line>"
git push -u origin $ARGUMENTS
just pr "<title>"
```

`just pr` uses a templated body (Summary + Test plan checklist). Paste the final report from step 6 into the Summary section before merging.

## Cleanup

After the PR lands:

```
just worktree-rm $ARGUMENTS
```

Terminates the mutagen session and removes the worktree on both mac and thor.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: task-spooler not installed` | tsp missing on thor | `just bootstrap-tsp` (one-time, needs sudo) |
| `ERROR: pgx-lower-dev not running` | container stopped | `just up` |
| `compile` queues but never runs | tsp daemon stuck | `ssh comfy 'TS_SOCKET=/tmp/pgx-build.sock tsp -K'` to kill daemon; it respawns on next use |
| Test passes that shouldn't | stale `.so` in PG | `just compile` re-installs the extension into the container's PG; rerun `just test` |
| Mutagen shows conflicts | simultaneous edit mac/thor | `mutagen sync flush pgx-lower-<slug>`; resolve manually, commit |
