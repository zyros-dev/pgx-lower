---
description: Autonomously review every open spec PR. Merge the clean ones; leave a PR comment on the rest. No args, no pauses, no user confirmation.
argument-hint: ""
allowed-tools: "Bash(just *) Bash(git *) Bash(gh *) Bash(ssh comfy *) Bash(cd *) Agent"
---

Review every open PR whose head branch matches `spec-NN-*`, oldest first.

Follow the `/merge` skill exactly. For each PR:

1. Spawn the `spec-reviewer` subagent for an independent review (background).
2. Binary decision: **merge** (clean `approve` + `MERGEABLE/CLEAN` + CI green) or **comment** (everything else). No middle ground.
3. Comment format: mirror the reviewer's structure (blocking / should-fix / optional / A/B sanity). Post via `gh pr comment`. Don't editorialize.
4. For the merge path only: after `gh pr merge --squash --delete-branch`, run `just spec-complete NN <PR>` and `just worktree-rm spec-NN-<keyword>` from the main checkout.
5. Move to the next PR. Do not pause for user input between PRs.

Spawn reviewers in parallel (background `run_in_background:true`) when you have multiple PRs — they're independent.

End with a final tally:
- `✓ #NN spec XX — merged`
- `✗ #NN spec XX — commented (one-line reason)`

If any PR got commented on, tell the user they can invoke `/pgx:read-pr-feedback` to see the accumulated feedback and decide whether to redispatch implementers.

Do not ask the user for anything — not which PRs to review, not whether to spawn the reviewer, not whether to merge. The binary merge-or-comment rule replaces the old pre-merge confirmation pause.
