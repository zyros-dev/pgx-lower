---
description: Review every open spec PR (one by one), handle conflicts, merge after one user confirmation per PR, mark spec done. No args.
argument-hint: ""
allowed-tools: "Bash(just *) Bash(git *) Bash(gh *) Bash(ssh comfy *) Bash(cd *)"
---

Review and (where appropriate) merge every open PR whose head branch matches `spec-NN-*`, oldest first.

Follow the `/merge` skill exactly. For each PR:

1. Spawn the `spec-reviewer` subagent for an independent review.
2. Act on the verdict per the skill's decision table.
3. Handle rebase/conflicts per the skill (try rebase; if conflicts, comment on the PR and stop on that PR — never auto-resolve).
4. Pause once for user confirmation immediately before `gh pr merge`. Do not pause anywhere else.
5. After merge: `just spec-complete NN <PR>` and `just worktree-rm spec-NN-<keyword>` from the main checkout.
6. Move to the next PR with a one-line summary ("✓ #42 spec 02 merged" / "✗ #45 spec 05 needs changes").

Do not ask the user which PRs to review, or whether to spawn the reviewer, or whether to rebase. The only mandatory pause is the per-PR pre-merge confirmation. End with a final tally across all PRs handled.
