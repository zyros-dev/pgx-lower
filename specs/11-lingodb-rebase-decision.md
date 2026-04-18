# Spec 11 — LingoDB rebase decision (decision spec, not implementation)

**Tier:** decision
**Stack on:** 09 + 10 results
**Blocks:** —
**Estimated effort:** 1 day to write decision; weeks to execute if "yes"

## Goal

Decide whether to re-vendor modern LingoDB (which has the SubOp dialect and
modernised vectorisation infrastructure) on top of our pgx-lower fork, or
keep the current snapshot indefinitely.

**Output is a written decision and a follow-up plan, not code.**

## Background

`include/lingodb/` and `src/lingodb/` contain a vendored snapshot of LingoDB
(~432KB headers + sources). The snapshot predates LingoDB's SubOp dialect.
Modern LingoDB uses SubOp as its batching/vectorisation foundation — the
intermediate dialect between RelAlg and DB+DSA where most of the SIMD-friendly
patterns live.

Our current pipeline (RelAlg → DB+DSA → Standard → LLVM) skips the SubOp
layer because we don't have it. Specs 09–10 build vectorisation on top of
the current dialect set instead.

The local tree has no version metadata or upstream commit hash. The
`lingo-db/` directory at the repo root is empty (likely intended as a
future submodule mount point).

## What this spec produces

A decision document at `docs/lingodb-rebase-decision.md` (create the
directory) with these sections:

### 1. State after specs 09+10

Describe what spec 09's vectorisation actually delivered. Concrete numbers
on q01, q06, q03, q12 — both vs spec 02 baseline and vs `main`.

If the wins met or exceeded the SubOp-dialect targets that motivated the
rebase consideration, that's strong evidence to **stay**.

### 2. SubOp inventory

What does upstream LingoDB's SubOp give us that we don't have? Read upstream
(via web fetch on github.com/lingo-db/lingo-db is fine) and list the dialect's
ops. Map each to either:
- "We already do this in our custom patterns from spec 09" (no value).
- "We don't have this and it'd be a clear win" (value).
- "We don't have this and don't need it" (no value).

### 3. Migration cost estimate

Concrete tasks to rebase:

- Identify the upstream LingoDB commit closest to (but not before) the SubOp
  introduction. Estimate diff size (`git log --shortstat` between our snapshot
  and that commit).
- List every file we modified locally (anything in `src/lingodb/` or
  `include/lingodb/` that diverges from upstream). Categorise each as:
  - Patch must be reapplied on top of upstream
  - Patch is now obsolete because upstream did the same thing
  - Patch fights upstream and must be redesigned
- Estimate the type-system-divergence work: our DB dialect was modified for
  PG-native dates (spec 06) and varlena strings (spec 07). Upstream may have
  evolved the same types differently.

### 4. Risk inventory

- How long has it been since we last validated the full TPC-H suite end-to-end?
- Does upstream LingoDB have a test suite we can run after rebase?
- What's the failure mode if SubOp lowering is buggy — does it break a single
  query type or the whole pipeline?

### 5. The decision

Pick one:

- **Stay (keep snapshot).** Document the rationale. List the upstream features
  we're explicitly choosing not to track. Set a calendar reminder for 6 months
  to re-evaluate.
- **Rebase.** Write a follow-up implementation spec (12+) with the migration
  plan from section 3 fleshed out into discrete tasks.
- **Cherry-pick.** Land specific upstream patches without a full rebase.
  Write a follow-up spec naming each cherry-picked commit and its rationale.

### 6. Concrete defaults

Default to **stay** unless one of these is true:

- Specs 09+10 fell short of their target by >50% AND SubOp would clearly
  close the gap.
- A specific feature we want for an upcoming spec only exists in modern
  LingoDB (e.g., parallel execution, distributed planning).
- We're spending more effort working around the snapshot's limitations than
  a rebase would cost.

The default-to-stay bias exists because: a 1.3MB fork rebase is a
multi-week yak shave that produces zero benchmark movement on its own.

## Acceptance criteria

- The decision document exists and answers all six sections above with
  concrete numbers and citations.
- A short summary (≤200 words) is added to the project memory at
  `~/.claude/projects/-Users-nickvandermerwe-repos-pgx-lower/memory/`
  via your normal memory-writing flow, so future sessions can find the
  decision without re-reading the doc.

## Not in scope

- Actually doing the rebase. That's a follow-up spec (numbered 12+) only
  written if section 5 picks "Rebase" or "Cherry-pick".

## A/B test

None — this is a decision, not code. The A/B numbers from specs 09+10 are
the inputs.
