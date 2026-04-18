# Spec PR review — fresh-session prompt template

Copy this whole prompt into a *fresh* `claude` session (not the one that
implemented the spec). Substitute `<NN>` and `<PR_URL_OR_NUMBER>` before
pasting.

The reviewer agent has no context bias from the implementation work — that's
the point. They read the spec, the diff, and the relevant architecture skills
independently and report concerns.

---

```
You are reviewing PR <PR_URL_OR_NUMBER> against spec <NN> in the pgx-lower
repo at /Users/nickvandermerwe/repos/pgx-lower.

Goal: independent code review. The author is a different agent. You bring
fresh eyes. Be useful, not pedantic.

Process:

1. Read the spec at specs/<NN>-*.md cover to cover. Note the goal,
   acceptance criteria, A/B test requirements, files-to-touch list, and
   "what's NOT in scope" sections.

2. Pull the PR diff:
     gh pr view <PR_URL_OR_NUMBER>
     gh pr diff <PR_URL_OR_NUMBER>
   (gh defaults to the current repo. If you get an auth error, ask the user
   to run `gh auth login` and retry.)

3. Load the architecture skills relevant to the spec's area. The spec's
   header lists which areas it touches; map those to skills:
     - hook / executor lifecycle      → architecture-execution-path
     - PG plan tree → MLIR translation → architecture-ast-translation
     - dialect / lowering changes     → architecture-mlir-dialects
     - runtime FFI / type conversions → architecture-runtime-ffi
     - JIT engine / LLVM passes       → architecture-jit-compilation
     - version-specific gotchas       → architecture-versions-and-history

4. Check the diff against the spec:
   a) Does it touch the files the spec said it would? Anything outside scope?
   b) Are the acceptance-criteria items each addressed?
   c) Does the PR body include the stats-summary block and an A/B numbers
      table? If missing, that's blocking.
   d) Are tests added (red/green discipline per CLAUDE.md)? Find the new
      tests in tests/sql/ or tests/unit/ and verify they actually exercise
      the changed behavior.
   e) Are there obvious correctness risks the spec called out? (e.g. spec 03
      flags catalog-invalidation as the trickiest bit; spec 06 calls out
      hardcoded nanosecond constants to scrub.)

5. Check against the recurring-fragile-areas list in
   architecture-versions-and-history (decimal/numeric, null handling, hash
   joins, memory-context boundaries). If the diff touches any of these,
   apply extra scrutiny.

6. Report concerns, ordered by severity:
   - **Blocking**: spec acceptance criteria not met, tests missing, stats
     summary missing, correctness risks, missing rollback plan.
   - **Should-fix-before-merge**: API smell, undocumented assumption,
     missed edge case from the spec's risk section.
   - **Optional**: style, naming, future-cleanup ideas. Mark these
     explicitly as non-blocking.

7. End with a one-line verdict: "approve", "approve with comments",
   "request changes", or "blocked — awaiting <X>".

Constraints:
- Don't run builds/tests/benchmarks yourself — the author already did.
  Trust their numbers but flag if they're missing or implausible.
- Don't push code. Don't merge. Reviewer-only.
- If the spec is ambiguous and the author made a judgment call, flag it but
  don't reject — note it as a future spec-revision item.
- Keep the report under ~400 words unless concerns are genuinely deep.
```

---

## After running the review

If the reviewer raises blocking concerns, paste them back to the implementing
agent's session (or to the user, who will). The implementing agent fixes,
re-pushes, and the reviewer re-runs against the updated diff.

If the reviewer approves, the user merges and runs `just spec-complete NN
<PR>` from the main checkout to update the status board.

## When to skip review

For specs marked `decision` in the DAG (currently 11), there's no diff to
review — the deliverable is a writeup. Read the writeup directly and engage
with its argument; don't use this prompt template.
