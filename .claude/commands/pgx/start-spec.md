---
description: Implement a spec from specs/ end to end (claim → worktree → TDD → PR → in_review). One arg: the spec number, e.g. /pgx:start-spec 03.
argument-hint: "<spec number, e.g. 03>"
allowed-tools: "Bash(just *) Bash(git *) Bash(gh *) Bash(ssh comfy *) Bash(mutagen *) Bash(cd *)"
---

Implement spec `$ARGUMENTS` from `specs/` end to end.

Follow the `/devops` skill exactly. Don't ask the user for branch names, slugs, worktree details, or permission to run recipes — derive sensible defaults from the spec filename and proceed. The user's only contract is naming the spec; everything else (claim, worktree, red/green TDD loop, build, check, bench, PR, in_review) is on you.

Escalate only when something genuinely blocks: claim conflict on the status board, build broken in a way you can't diagnose, spec genuinely ambiguous, or unrelated test failures.
