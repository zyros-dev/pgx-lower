# pgx-lower — always-loaded context

## Where things run

**Everything runs on thor.** The mac is an edit host only. Don't build, run Postgres, execute the extension, or run benchmarks locally — the toolchain (LLVM 20, MLIR 20, Postgres 17.6 from source) lives in a Docker image on thor.

Edits sync to thor via a **mutagen** session. Main repo: session `pgx-lower` between `/Users/nickvandermerwe/repos/pgx-lower` (mac, alpha) and `comfy:/home/zel/repos/pgx-lower` (thor, beta). Worktrees get their own sessions (`pgx-lower-<slug>`), managed by `just worktree-new`. Give mutagen a second after editing a file before running a command on thor.

Thor SSH alias: `comfy` (user `zel`; see `~/repos/midgard/docs/infrastructure.md`).

## How to do work

For any code change that ends in a PR, follow the **`/devops` skill** (`.claude/skills/devops/SKILL.md`). It covers worktree → test → implement → build → check → bench → report → PR, all via `justfile` recipes that queue through `tsp` on thor so concurrent agents don't collide. Run `just --list` for the recipe surface.

**The user only types short trigger phrases.** Everything else is on the agent.

| Trigger | Skill | What you do |
|---------|-------|-------------|
| `start spec NN` / `implement spec NN` | `/devops` | Claim, worktree, TDD loop, open PR, mark in_review |
| `merge spec NN` / `review pending PRs` / `merge ready PRs` | `/merge` | Spawn `spec-reviewer` subagent, handle conflicts, confirm with user once, merge, mark done, remove worktree |
| `spec NN merged` (manual case) | inline | `just spec-complete NN <PR>` + `just worktree-rm <slug>` |

Don't ask the user for branch names, slugs, PR numbers you can derive, worktree details, or permission to run recipes. Pick sensible defaults from the spec filename and proceed. Escalate only when something genuinely blocks (claim conflict, build broken in a way you can't diagnose, spec ambiguous, conflicts on rebase). The merge skill has one mandatory user-confirmation pause: just before `gh pr merge`. Everything else is autonomous.

## Red/green TDD is required

Write the failing test **first**. Run `just test` and confirm the expected failure before touching any implementation. Implement the minimum change to turn it green. Only then refactor. No exceptions — "I'll add the test after" produces untested code and we don't merge untested code.

## Specs

Performance-roadmap specs live in `specs/`. **`specs/00-dag.md`** defines what
can be worked on and in what order. **`specs/STATUS.md`** is the live board of
which specs are claimed / in progress / done — read it before starting any
spec work. Update via `just spec-claim NN BRANCH`, `just spec-in-review NN PR`,
`just spec-complete NN PR`. When working from a spec, the spec file is the
authoritative description of what to build; the devops skill walks the
implement-and-ship loop.

Until you're given a spec, work from the user's task description.

## Skill catalog

Skills are the primary knowledge layer. Each is loaded on demand.

- **`/devops`** — implement-and-ship workflow (worktree → red → green → check → bench → PR).
- **`/merge`** — review-and-merge workflow for spec PRs (spawns spec-reviewer subagent, handles conflicts, merges).
- **`/architecture-overview`** — top-level map; entry point if you're disoriented.
- **`/architecture-execution-path`** — PG executor hook → MLIR runner → JIT chain.
- **`/architecture-ast-translation`** — PG plan tree → MLIR RelAlg.
- **`/architecture-mlir-dialects`** — RelAlg / DB / DSA / util + the lowering pipeline.
- **`/architecture-runtime-ffi`** — C runtime called from JITed code (hashtables, sort, type conversions).
- **`/architecture-jit-compilation`** — LLVM JIT engine, optimization, ExecutionEngine.
- **`/architecture-versions-and-history`** — LLVM 20 / MLIR 20 / PG 17.6 pinning, gotchas, lessons from history.
