# Spec status board

Single pane of glass for the performance-roadmap specs in this directory. The
DAG (`00-dag.md`) defines *what can be worked on*; this file tracks *what's
actually being worked on right now*.

Update via `just spec-claim NN BRANCH`, `just spec-release NN`, `just spec-complete NN PR`.
Don't edit the table by hand — the recipes preserve column alignment and commit
atomically against `main` so concurrent agents can't claim the same spec.

## States

- `available` — no one is working on it; blockers (per the DAG) may still apply.
- `claimed` — someone is starting; branch may not exist yet.
- `in_progress` — branch exists, work is happening.
- `in_review` — PR is open, awaiting review/merge.
- `blocked` — work paused; see Notes for why.
- `done` — landed on `main`. Don't claim.

## Board

| Spec | State | Owner | Branch | PR | Notes |
|------|-------|-------|--------|----|-------|
| 01-hoist-mlir-context | available | — | — | — | |
| 02-enable-llvm-vectorization | available | — | — | — | |
| 03-plan-shape-compile-cache | available | — | — | — | blocked by 01 per DAG |
| 04-cost-gate-small-queries | available | — | — | — | blocked by 03 per DAG |
| 05-decode-at-scan | available | — | — | — | |
| 06-pg-native-date-repr | available | — | — | — | blocked by 05 per DAG |
| 07-pg-varlena-strings | available | — | — | — | blocked by 05 per DAG |
| 08-inline-pg-bitcode | available | — | — | — | blocked by 05+06+07 per DAG |
| 09-mlir-vector-dialect | available | — | — | — | scope informed by 02 result |
| 10-row-to-column-transpose | available | — | — | — | paired with 09 |
| 11-lingodb-rebase-decision | available | — | — | — | decision spec, not code |
| 12-copy-and-patch-backend | available | — | — | — | deferred until 01–04 land |
| 13-adaptive-execution | available | — | — | — | deferred until 09 lands |
| 14-add-version-function | available |  —  |  —  |  —  | smoke spec for harness validation; not roadmap work |

## Conventions

- **Owner** is a free-form short identifier. Use whatever helps you tell agents
  apart at a glance (e.g. `claude-04-19-am`, `nick-pair`, `claude-fast-mode`).
- **Branch** matches the worktree name from `just worktree-new`.
- **PR** is the GitHub PR number once one is open.
- **Notes** is for blockers, partial progress, "running A/B now," etc.
- A `done` row stays on the board so the next agent can scan landed work
  without diffing git history.

## Reading the DAG together with this file

A spec being `available` here doesn't mean you can start it — its row in
`00-dag.md` may list a blocker that's still `available` or `in_progress`.
Always check the DAG first.
