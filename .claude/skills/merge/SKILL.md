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
whether to spawn the reviewer, don't ask permission to rebase. The only
mandatory pause is **before merging** — get explicit confirmation, since
merging is hard to reverse.

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

## 3. Decide

Branch on the verdict:

| Verdict | Action |
|---------|--------|
| `approve` | Proceed to step 4. |
| `approve with comments` | Post the optional/future items as a single PR comment via `gh pr comment <PR> --body "<text>"`, then proceed to step 4. |
| `request changes` | Post the blocking + should-fix items as a PR review (`gh pr review <PR> --request-changes --body "<text>"`). Mark the spec back to `in_progress` via `just spec-claim NN <slug>` (forces a re-claim signal). Tell the user; do not merge. |
| `blocked — awaiting X` | Post a comment explaining what's blocked. Run `just spec-block NN "<reason>"`. Tell the user. |

If the subagent's verdict is `approve` but you spot something it missed
that's clearly blocking (e.g. CI red, mergeable=CONFLICTING with a clearly
broken rebase), override and treat as `request changes`. Cite the specific
evidence.

## 4. Pre-merge: handle conflicts

```
gh pr view <PR> --json mergeable,mergeStateStatus -q '.mergeable + " / " + .mergeStateStatus'
```

| State | Meaning | Action |
|-------|---------|--------|
| `MERGEABLE / CLEAN` | Ready. | Skip to step 5. |
| `MERGEABLE / UNSTABLE` | Mergeable but CI failing. | Stop. Don't merge against red CI. Tell user. |
| `MERGEABLE / BLOCKED` | Required reviews missing. | Stop. Tell user (they may need to add a review). |
| `CONFLICTING / DIRTY` | Conflicts with main. | Try a rebase — see below. |
| `UNKNOWN / *` | GitHub hasn't computed yet. | Wait 5s and re-check once. If still unknown, tell user. |

### Rebase attempt

```
git fetch origin
git checkout <head-branch>
git rebase origin/main
```

- **Clean rebase**: `git push --force-with-lease`, then re-check mergeability. If now `CLEAN`, proceed.
- **Conflicts**: do not auto-resolve. Stop, run `git rebase --abort`, post a PR comment explaining where conflicts are, and tell the user. The implementing agent (or user) handles the resolution.

## 5. Confirm with user before merging

This is the only mandatory pause. Show the user:
- PR number, title, branch.
- Reviewer verdict (one line).
- Mergeability state.
- Any optional concerns the reviewer noted.

Wait for "yes", "merge", "go ahead", or equivalent. Anything ambiguous → ask again. "no" or "wait" → stop.

## 6. Merge

```
gh pr merge <PR> --squash --delete-branch
```

Use squash by default (matches the convention in the existing repo
history — single coherent commit per spec). Switch to `--merge` only if
the user asks for a merge commit.

`--delete-branch` cleans up the remote branch.

## 7. Post-merge: status board + worktree cleanup

Capture the PR number you just merged. Then from the **main checkout**:

```
cd ~/repos/pgx-lower
git pull --rebase origin main
just spec-complete NN <PR>
just worktree-rm spec-NN-<keyword>
```

If `worktree-rm` fails because the worktree doesn't exist on this machine
(e.g. another agent created it), that's fine — it'll error harmlessly.

## 8. Loop or stop

If the trigger was `review pending PRs` / `merge ready PRs` and there are
more open PRs, return to step 2 with the next one. Tell the user a
one-line summary per PR as you go ("✓ #42 spec 02 merged, ✗ #45 spec 05
needs changes").

If the trigger was `merge spec NN`, you're done. Report the final state.

## When things go wrong

| Situation | What you do |
|-----------|-------------|
| `gh` auth error | Stop, tell user to run `gh auth login`. Don't try to authenticate. |
| Spec NN has no open PR | Tell user; don't invent one. |
| Multiple open PRs match `spec-NN-*` | Stop, ask user which. |
| spec-reviewer subagent times out / errors | Try once more. If it fails again, tell user; don't fall back to your own review. |
| CI is red and reviewer missed it | Override to `request changes` per step 3. |
| Rebase produces conflicts | `git rebase --abort`, comment on PR, tell user. Never auto-resolve. |
| `gh pr merge` fails with "branch protection" | Tell user; don't try to bypass. |
| `just spec-complete` fails | Probably push race. Re-run once. If still failing, tell user. |
| User says "merge ready PRs" but none qualify | Tell them what's open and why each is disqualified. |

## What this skill does NOT do

- It does not re-implement the reviewer's logic. The subagent is canonical.
- It does not approve PRs in GitHub's UI on the user's behalf (`gh pr review --approve`). Skip that step; the merge itself is the approval signal.
- It does not handle non-spec PRs. Those go through manual review.
- It does not modify code. Only `git rebase` (which is recoverable) and `gh pr merge` (which is not — hence the mandatory pause).
