---
description: Show accumulated review feedback across every open spec PR. Groups by PR, shows the most recent spec-reviewer comment, highlights what the implementer would need to re-do. No args.
argument-hint: ""
allowed-tools: "Bash(gh *) Bash(git *)"
---

For every open PR whose head branch matches `spec-NN-*`, list the accumulated review feedback so the user can decide whether to redispatch implementers.

Steps:

1. `gh pr list --state open --json number,title,headRefName --limit 50` to enumerate open spec PRs.
2. For each matching PR, in oldest-first order:
   - Fetch the PR's issue-style comments: `gh api repos/zyros-dev/pgx-lower/issues/<PR>/comments --jq '.[] | {user: .user.login, created_at, body}'`.
   - Filter to comments that start with `Independent review (spec-reviewer subagent)` — those are the ones the `/merge` loop posted. Keep only the most recent such comment per PR (older reviews are superseded).
   - Also fetch review-level comments if any: `gh pr view <PR> --json reviews -q '.reviews[] | select(.state == "CHANGES_REQUESTED") | {author: .author.login, submittedAt, body}'`.
3. For each PR, output a block:
   ```
   ## #<PR> — spec <NN> — <title>
   Branch: spec-NN-<slug>
   Mergeability: <MERGEABLE / CLEAN | ...>
   CI: <pass/fail summary>

   ### Most recent reviewer feedback (<date>)
   <full comment body>
   ```
4. At the end, list a short action menu:
   - PRs that need the implementer to fix should-fix items
   - PRs that are blocked on a rebase (CONFLICTING)
   - PRs where CI is red
   - PRs that have no reviewer comment yet (run `/pgx:review-open-prs` first)

5. If the user asks for a redispatch, do NOT act on that automatically — route them to `/pgx:start-spec NN` or ask whether they want you to spawn a fixer subagent inline.

Don't truncate the reviewer's comment bodies — the point of this command is to surface the feedback so the user can read it. If a single PR has many superseded reviews, only show the most recent to keep the output scannable, but mention how many older ones exist.
