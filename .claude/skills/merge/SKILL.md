---
name: merge
description: End-to-end review-and-merge workflow for spec PRs in pgx-lower. Auto-invoke when the user says "merge spec NN", "review pending PRs", "merge ready PRs", or "review and merge". Spawns the spec-reviewer subagent for an independent review, handles rebase/conflict resolution, runs the merge, and updates the spec status board. The user only ever names the trigger — every other step is yours.
disable-model-invocation: false
argument-hint: "<spec number, e.g. 03> OR 'all' for every open PR"
allowed-tools: "Bash(just *) Bash(git *) Bash(gh *) Bash(ssh comfy *) Bash(cd *)"
---

# merge — review-and-merge playbook

Counterpart to `/devops` (which implements). This skill closes the loop:
review the resulting PR with independent eyes, handle conflicts, merge,
update the status board.

**The user does not orchestrate.** They say "merge spec NN" or "review
pending PRs" and you do every step below. Don't ask which PR, don't ask
whether to spawn the reviewer, don't ask permission to rebase, don't
pause before merge. Every PR ends in one of two terminal actions:

1. **Merge** — clean approval, no blocking concerns, no should-fix items.
2. **Comment** — any hesitation at all (should-fix items, spec drift,
   unticked test-plan boxes, missing microbench, mergeability issues,
   CI failures, rebase conflicts). Post the full reviewer concerns as a
   PR comment, leave the PR open, move on.

There is no middle ground, no "approve with comments then merge", no
"merge after user confirms the should-fix is acceptable". The user
redispatches work later via `/pgx:read-pr-feedback` — your job is to
get the review signal onto each PR cleanly and move to the next one.

## 0. Preflight

```
cd ~/repos/pgx-lower      # main checkout
git fetch origin --prune
just spec-status
gh pr list --state open --json number,title,headRefName,mergeable,statusCheckRollup
```

Now you know what's open and which specs they correspond to.

## 1. Pick the PR(s) to handle

Mapping rules:
- `merge spec NN` → find the open PR whose head branch starts with
  `spec-NN-`. If multiple, error and ask the user.
- `review pending PRs` / `review and merge` → loop through *all* open PRs
  whose head branch starts with `spec-`, oldest first.
- `merge ready PRs` → only PRs that are `mergeable=MERGEABLE` AND CI is
  passing AND already have an `approved` review.

Skip:
- Drafts (`isDraft = true`).
- PRs whose head branch doesn't match `spec-NN-*` (those aren't spec PRs;
  this skill isn't the right fit — tell the user).

## 2. Independent review

For each PR, spawn the `spec-reviewer` subagent via the Agent tool. Pass:
- The spec ID (parsed from the branch name `spec-NN-*`).
- The PR number.

Subagent returns a structured report. Read the verdict + concerns. Do not
re-do the review yourself — your job is to act on the verdict, not
duplicate it.

## 3. Decide: merge or comment

Binary decision. No middle ground.

**Merge** (proceed to step 4) when ALL of these hold:
- Reviewer verdict is `approve` with no should-fix items listed.
- Mergeability is `MERGEABLE / CLEAN`.
- Every CI check reports `SUCCESS` (or no blocking checks are configured).

**Comment and move on** (skip to step 5) for every other case:
- Reviewer returned `approve with comments`, `request changes`, or
  `blocked` → comment with the full concern list.
- Reviewer said `approve` but you spot something it missed that's
  clearly blocking (CI red, `CONFLICTING / DIRTY`, merge-state
  `BLOCKED`/`UNSTABLE`) → comment citing the evidence and override to
  comment-and-move-on. Don't merge over a clearly-broken state.
- `UNKNOWN / *` mergeability → wait 5s, re-check once; if still
  unknown, comment ("GitHub hasn't computed mergeability — check back
  shortly") and move on.

The comment format is:

```
gh pr comment <PR> --body "$(cat <<'EOF'
Independent review (spec-reviewer subagent) — verdict: <verdict>.

## Blocking
<items or "none">

## Should-fix before merge
<items or "none">

## Optional / future
<items or "none">

## A/B sanity
<one paragraph>
EOF
)"
```

Mirror the structure the reviewer produced. Don't editorialize. If the
reviewer flagged a concern, it goes in the comment even if you think
it's minor — the user sorts priority via `/pgx:read-pr-feedback`.

Additionally:
- If the reviewer returned `request changes`, also run `just
  spec-claim NN <slug>` to flip the board back to `in_progress` (this
  nudges the status board so future agents see re-work is needed).
- If the reviewer returned `blocked — awaiting X`, run `just
  spec-block NN "<reason>"`.

## 4. Rebase before merging (merge path only)

If the merge path was selected in step 3 and mergeability is already
`CLEAN`, skip this step.

If mergeability is `CONFLICTING / DIRTY` but the reviewer gave a clean
`approve`: don't try to rebase. Comment-and-move-on with a rebase
request — the human judgment needed to resolve conflicts belongs on
the implementing agent, not on this loop.

(The old "try a rebase here" path is gone. Rebase conflicts are now
always a comment-and-move-on signal, which matches the binary
merge-or-comment rule.)

## 5. Merge (merge path only)

```
gh pr merge <PR> --squash --delete-branch
```

Use squash by default (matches the convention in the existing repo
history — single coherent commit per spec). Switch to `--merge` only if
the user asks for a merge commit.

`--delete-branch` cleans up the remote branch.

## 6. Post-merge: status board + worktree cleanup

Capture the PR number you just merged. Then from the **main checkout**:

```
cd ~/repos/pgx-lower
git pull --rebase origin main
just spec-complete NN <PR>
just worktree-rm spec-NN-<keyword>
```

If `worktree-rm` fails because the worktree doesn't exist on this machine
(e.g. another agent created it), that's fine — it'll error harmlessly.

## 7. Loop or stop

If the trigger was `review pending PRs` / `merge ready PRs` and there
are more open PRs, return to step 2 with the next one. Don't pause
between PRs — the binary merge-or-comment rule gives every PR a
terminal action without user input. At the end, tell the user a
one-line tally:

```
✓ #42 spec 02 — merged
✗ #45 spec 05 — commented (fragile struct layout, unticked test plan)
✗ #48 spec 07 — commented (MERGEABLE/UNSTABLE, CI red on utest)
```

Point them at `/pgx:read-pr-feedback` if any PR got commented on.

If the trigger was `merge spec NN`, you're done. Report the final
state.

## When things go wrong

| Situation | What you do |
|-----------|-------------|
| `gh` auth error | Stop, tell user to run `gh auth login`. Don't try to authenticate. |
| Spec NN has no open PR | Tell user; don't invent one. |
| Multiple open PRs match `spec-NN-*` | Stop, ask user which. |
| spec-reviewer subagent times out / errors | Try once more. If it fails again, post a PR comment saying "automated review failed twice — human review needed" and move on. Don't fall back to your own review. |
| CI is red and reviewer missed it | Override to comment-and-move-on. Cite the failing check by name in the PR comment. |
| Mergeability is `CONFLICTING / DIRTY` | Comment on the PR with a rebase request. Don't auto-rebase — the binary merge-or-comment rule removes the old try-a-rebase path. |
| `gh pr merge` fails with "branch protection" | Post a PR comment explaining the protection rule hit and tell the user; don't try to bypass. |
| `just spec-complete` fails | Probably push race. Re-run once. If still failing, tell user. |
| User says "merge ready PRs" but none qualify | Tell them what's open and that each got a PR comment instead. |

## What this skill does NOT do

- It does not re-implement the reviewer's logic. The subagent is canonical.
- It does not approve PRs in GitHub's UI on the user's behalf (`gh pr review --approve`). Skip that step; the merge itself is the approval signal.
- It does not handle non-spec PRs. Those go through manual review.
- It does not modify code. Only `gh pr merge` (which is irreversible, but now gated by the binary rule: only truly clean PRs get merged, and anything less gets a comment).
- It does not pause for user confirmation. The reviewer subagent is the second pair of eyes; the binary merge-or-comment rule is the safety rail.
