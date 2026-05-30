# pgx-lower — always-loaded context

## Four rules (apply every code write)

1. **Think Before Coding.** No silent assumptions. State assumptions. Surface tradeoffs. Ask before guessing. Push back when a simpler approach exists.
2. **Simplicity First.** Minimum code that solves the problem. No speculative features. No abstractions for single-use code.
3. **Surgical Changes.** Touch only what you must. Don't improve adjacent code, comments, or formatting. Match existing style.
4. **Goal-Driven Execution.** Define success criteria. Loop until verified.

Comments: would a human write this? If a comment restates what well-named code says, delete it. File-header banners are noise unless they encode a non-obvious WHY. In tests, the test names ARE the documentation.

## Where things run

**Everything runs on thor.** The mac is an edit host only. Don't build, run Postgres, execute the extension, or run benchmarks locally — the toolchain (LLVM 20, MLIR 20, Postgres 17.6 from source) lives in a Docker image on thor.

Edits sync to thor via a single **mutagen** session named `pgx-lower`, between `/Users/nickvandermerwe/repos/pgx-lower` (mac, alpha) and `comfy:/home/zel/repos/pgx-lower` (thor, beta). We work in this one checkout — no worktrees. Feature work is a branch in this directory; the recipes always flush the `pgx-lower` session, so any branch's edits reach thor. Give mutagen a second after editing before running a command on thor (the recipes' `_preflight` flush handles this for you). If the session's ignore list drifts, `just sync-main-reset`.

Thor SSH alias: `comfy` (user `zel`; see `~/repos/midgard/docs/infrastructure.md`).

## How we work: spec-first, human-in-the-loop

Specs are **curated artifacts**, authored with the user, that live in the wiki and outlive any one PR. The user shapes them until they're ready, then triggers implementation. Nothing is autonomous — the user is in the loop for every spec.

The two durable artifacts live at `~/repos/sandbox/wiki/specs/pgx-lower/`:

```
designs/  YYYY-MM-DD-<topic>-design.md   # brainstorming output; being shaped, iterative
plans/    YYYY-MM-DD-<topic>-plan.md     # writing-plans output; ready-to-execute = the work queue
```

The lifecycle, driven by the **superpowers** skills:

1. **Design** — user says "let's spec X" → `brainstorming` skill → design file in the wiki. Refined across as many sessions as the user wants.
2. **Plan** — "plan it" → `writing-plans` skill → plan file in the wiki. The user can stack several plans across their own time.
3. **Implement** — "implement the ready plans" → work the queue **sequentially** in this checkout: a branch + PR per plan, each gated by that plan's own acceptance criteria. PRs exist so the user reads diffs in Gitea/GitHub instead of opening CLion.

Commit a wiki spec/plan from `~/repos/sandbox/`: `git add wiki/specs/ && git commit -m "specs: pgx-lower — <topic>"` (the sandbox cron pushes; don't push manually).

## Per-plan gates (declared in the plan, not global)

Default bar for every plan, before its PR opens:

- **red/green TDD** — write the failing test first, run `just test`, confirm it fails, then implement the minimum to turn it green. No exceptions; we don't merge untested code.
- `just check-diff` clean on touched files.
- `just compile` + `just utest` + `just test` green (this includes the fast TPC-H-as-correctness regression checks — run a query, diff output vs stock PG).

Opt-in, only when the plan's "Done means" turns it on:

- `just bench` + `just bench-report` — the A/B speed report. Token-heavy (full TPC-H sweep on thor); most plans don't need it. Correctness ≠ benchmark: validate correctness always, benchmark only to prove a speedup the plan promised.

Run `just --list` for the full recipe surface.

## Long-running subagents: run in the background

When spawning a subagent that will take more than a few minutes (implementing a plan, anything that runs `just compile` / `just bench` / `just test`), **pass `run_in_background: true`** to the Agent tool. A foreground agent blocks this conversation, and if the user submits a message (or Ctrl-C's) while it's running, the subagent dies mid-flight and we lose the work. Background agents notify on completion and survive interjections. Only run foreground if the result is needed within ~60s to decide the very next tool call.

## Skill catalog

Skills are the primary knowledge layer. Each is loaded on demand.

The spec workflow uses the **superpowers** skills: `brainstorming` (design), `writing-plans` (plan), `executing-plans` / `subagent-driven-development` (implement), `test-driven-development` (the red/green discipline).

Project architecture references:

- **`/architecture-overview`** — top-level map; entry point if you're disoriented.
- **`/architecture-execution-path`** — PG executor hook → MLIR runner → JIT chain.
- **`/architecture-ast-translation`** — PG plan tree → MLIR RelAlg.
- **`/architecture-mlir-dialects`** — RelAlg / DB / DSA / util + the lowering pipeline.
- **`/architecture-runtime-ffi`** — C runtime called from JITed code (hashtables, sort, type conversions).
- **`/architecture-jit-compilation`** — LLVM JIT engine, optimization, ExecutionEngine.
- **`/architecture-versions-and-history`** — LLVM 20 / MLIR 20 / PG 17.6 pinning, gotchas, lessons from history.
