---
name: strengthen-harness
description: Autonomous harness-improvement loop. Given a spec (the canary), launches a subagent to implement it under the devops flow, collects that agent's reflection on harness friction, closes the canary PR (not the point), launches a second subagent to fix the harness issues, reviews + merges that fix PR, then loops. Stops when two consecutive canary runs complete without substantive harness complaints. Designed to run unattended for hours.
argument-hint: "<canary spec number, e.g. 14>"
disable-model-invocation: false
allowed-tools: "Bash(just *) Bash(git *) Bash(gh *) Bash(ssh comfy *) Bash(mutagen *) Bash(cd *) Agent"
---

# strengthen-harness — autonomous harness-improvement loop

Runs a tight back-and-forth between two subagents until the harness stops producing complaints. You (the coordinator) never implement anything yourself — you spawn agents, review their PRs, decide merge-vs-close, and drive the loop.

## Contract with the user

- **Input:** one spec number (the canary). `/strengthen-harness 14` uses spec 14.
- **Output:** zero or more harness-fix PRs merged to main. Zero spec implementations merged (the canary is reset every loop so future runs start from the same clean state).
- **Termination:** two consecutive canary runs return "buttery" reflections — complaints are empty or contain only minor nits (typos, docs phrasing preferences). You decide what "buttery" means; err on the side of "one more loop" if it's borderline.
- **Time budget:** run as long as you need. A single loop is ~30-60 min (canary implementation + optional harness fix). Keep looping until the termination condition fires.

**The user is not in the room.** Don't stop to ask questions unless you genuinely can't proceed (e.g., a subagent raises an issue that isn't a harness problem and isn't resolvable autonomously). Every loop produces either a merged harness-fix PR or a decision "no fix needed, loop again."

## Loop structure

Each iteration has four phases. Do all four before deciding whether to terminate.

### Phase A: Canary run (spawn implementer subagent)

Spawn a subagent with the general-purpose agent type. Brief them:

- They are implementing spec `<NN>` end-to-end using the `/devops` skill.
- They follow the normal devops flow: claim, worktree, TDD, PR, in_review.
- **Crucial:** at the end, they must append a reflection block to their final message with this exact prompt format:

```
Reflect on running spec NN. What was hard, ambiguous, broken, or slower
than expected in the workflow? Specifically: any recipe that didn't behave
as the skill described, any step where you had to ask the user something
the skill should have answered, any place where the spec was unclear,
any tool error that needed working around. Be specific — file paths,
recipe names, exact error messages. Don't sugar-coat — the harness is
what gets fixed, not your work.
```

- They must open a PR and mark it in_review, but they do **not** close or merge it (that's your job).
- Give them full autonomy within the devops skill. Don't pre-bless recovery paths — if they hit claim conflicts, they follow the skill's spec-abandon guidance.

Collect their response. If they completed the canary, you have a PR number + a reflection. If they stopped early (couldn't proceed), treat that as a critical harness failure — extract the failure mode, skip phase B, and go straight to phase C to fix it.

### Phase B: Canary review + close

Read the subagent's code diff (`gh pr view <PR> --json files,additions,deletions`, `gh pr diff <PR>`). You're checking:

- Did they implement what the spec asked? (Sanity check only — don't second-guess design.)
- Is the diff *only* what the spec prescribed? If they strayed into the harness itself (editing `justfile`/skill) during a canary run, that's usable harness-fix content — note it for phase C, but still close the canary PR.

Then close the canary PR:

```
gh pr close <PR> --comment "Canary run for strengthen-harness loop — code not for merge. Reflection absorbed; any harness fixes will land in a separate PR."
gh pr edit <PR> --title "[canary] spec NN — strengthen-harness loop"
just spec-abandon NN "strengthen-harness canary, PR #<N>"
```

`spec-abandon` resets state so the next loop starts clean. **Do not skip it.**

### Phase C: Triage the reflection

Read the subagent's reflection. Bucket every complaint into:

1. **Genuinely broken / wrong** — stale docs, recipes that don't behave as documented, tool errors the agent had to work around. These go into the fix queue.
2. **Ambiguous / required interpretation** — skill guidance the agent had to translate. These also go into the fix queue if the translation risks getting it wrong next time.
3. **Editor / IDE friction** — clangd diagnostics, local-side ergonomics. Usually low-priority; note but don't necessarily act.
4. **Minor / nit** — phrasing preferences, taste calls. Ignore for termination-counting purposes but include if phase D is already happening.

If bucket 1 + bucket 2 is empty, this run was buttery. Increment the "consecutive clean" counter. If the counter hits 2, terminate (write the summary, exit).

If bucket 1 + bucket 2 has at least one item, reset the counter to 0 and go to phase D.

### Phase D: Spawn harness-fixer subagent

Spawn a second subagent with the general-purpose agent type. Brief them:

- They are fixing specific harness issues. List the fix-queue items verbatim — do not editorialize.
- They follow the `/devops` skill end-to-end: create a worktree, do the changes as TDD where applicable, open a PR, mark in_review (if there's a spec wrapper — most harness fixes won't have one, so skip the status-board update).
- **For non-spec branches** (the typical case): tell them to use a slug like `harness-fixes-<N>` or `fix-<issue-slug>` and skip spec-claim.
- They must open a PR but not merge it (that's your job).

When they return, you have a PR number.

### Phase E: Harness-fix review + merge

Read their diff. Check:

- Does it address every item from the fix queue?
- Are the edits scoped? (No drive-by refactors unless they're cleaning up their own mess.)
- Do recipes still parse? (`just --list` in the worktree — they should have run this themselves.)
- Did `just utest` / `just test` / `just compile` stay green? (They should have exercised all three in the devops flow.)

If the diff looks good, merge it:

```
gh pr merge <PR> --squash --delete-branch
just spec-complete <NN> <PR>   # only if it happens to be a spec PR
# or otherwise:
just worktree-rm <slug>
```

If the diff has problems, **leave a review comment and reject** — do NOT make the fixes yourself:

```
gh pr review <PR> --request-changes --body "..."
gh pr edit <PR> --title "[changes requested] <original title>"
```

Then spawn a follow-up subagent briefed with the specific changes needed. The loop keeps iterating until the fix is clean enough to merge. You do not implement the fix yourself — that defeats the autonomous-loop purpose. The only exception: if the fix subagent gets stuck in a genuinely-ambiguous design question, resolve it and write a short clarifying comment, then spawn a new agent.

### Loop back

After a harness-fix PR merges, pull main locally, then go back to phase A with a fresh canary run.

## Subagent-briefing template (phase A canary)

Copy-paste this verbatim into the Agent tool's `prompt` field. Substitute `$ARGUMENTS` for the spec number.

```
Implement spec $ARGUMENTS using the /devops skill end-to-end. You have
full autonomy within the skill — claim, worktree, TDD loop, compile,
check-diff, utest, test, bench, bench-report, PR, spec-in-review.

If the spec appears already-claimed, use just spec-abandon as per the
skill. The user has pre-authorized abandoning any prior state for this
spec.

When you're done (PR open, spec in_review), append a reflection block
as the FINAL section of your response, using this exact prompt:

  Reflect on running spec $ARGUMENTS. What was hard, ambiguous, broken,
  or slower than expected in the workflow? Specifically: any recipe
  that didn't behave as the skill described, any step where you had to
  ask the user something the skill should have answered, any place
  where the spec was unclear, any tool error that needed working
  around. Be specific — file paths, recipe names, exact error
  messages. Don't sugar-coat — the harness is what gets fixed, not
  your work.

Output format at the end of your message:
  ## PR
  <PR URL>

  ## Reflection
  <your numbered list of harness issues, grouped by severity>

Do NOT close or merge your own PR. Return control after opening it +
marking in_review + appending the reflection.
```

## Subagent-briefing template (phase D harness fixer)

```
Fix the following harness issues surfaced by a canary run of spec $SPEC.
Do not attempt to re-implement the canary spec itself — that's a
different agent's job. Your scope is ONLY the harness (justfile,
.claude/skills/, .claude/commands/, scripts/, specs/ docs, test infra).

Fix queue:
<numbered list of items from phase C bucket 1 + bucket 2>

Follow the /devops skill's worktree→PR flow. Branch name: harness-fixes-N
where N is one higher than the existing harness-fixes-<N> in the repo's
merged PR history. Skip spec-claim since this isn't a spec.

Exercise just compile, just utest, just test, just bench, just bench-
report as part of your devops flow. Open the PR and append a short
summary of what you changed mapped to each fix-queue item. Do NOT merge.
Return control after opening the PR + marking it ready.
```

## Termination summary template

When you hit two consecutive buttery runs, write:

```
## strengthen-harness: terminated

Canary spec: $ARGUMENTS
Loops completed: <N>
Harness-fix PRs merged: <list of PR numbers>
Final two canary PRs (closed, not merged): <list of PR numbers>

Last two reflections are attached below for audit. Both had
bucket-1 + bucket-2 complaints empty; the harness is now at a
steady state for this canary.

### Merged harness-fix summary
<one-line per PR: what it fixed>

### Last reflection
<last subagent's reflection block, verbatim>
```

Then stop. Do not spawn more subagents.

## Budget escape hatch

If 10+ loops run without convergence, stop and write a diagnostic:

```
## strengthen-harness: escalating

10 loops did not converge. This usually means one of:
  a) A complaint keeps being "fixed" but regenerating in a different form.
  b) The canary spec is pulling on a genuinely-hard edge of the harness
     that needs architectural change, not an iterative fix.
  c) Subagents are pattern-matching instead of reading the actual issue.

The complaint pattern across loops was: <common theme>

Suggest: <one or two concrete paths forward for the user>.
```

Do not silently continue past 10 loops. 2 hours of subagent churn on a bad signal is a lot of wasted compute.

## Design notes

- **Why close canary PRs instead of merging?** The canary is a harness probe, not a feature. If you merged every canary run, spec 14 would be implemented 30 times. The canary IS the harness stress test; resetting it each loop preserves its value.
- **Why spawn a second subagent for fixes instead of doing it yourself?** Two reasons: (1) keeps the loop autonomous and self-similar — same kind of agent doing same kind of work; (2) tests that the harness skill is usable by agents who weren't present for the complaint-gathering. If the fixer agent can't understand the fix queue, that's a harness documentation problem itself.
- **Why review their diff instead of auto-merging?** Because fixer agents occasionally "fix" by deleting the thing that complained, or by papering over a structural issue with a band-aid. Review catches that.
- **Why two-in-a-row instead of one?** One clean run might be coincidence (subagent overlooked a complaint). Two in a row means the harness is actually steady.
