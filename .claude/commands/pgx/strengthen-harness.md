---
description: Autonomous harness-improvement loop. Spawns a canary subagent to implement the given spec, collects their reflection on harness friction, closes the canary PR, spawns a fixer subagent to address the friction, merges the fix, and loops until two consecutive canary runs are "buttery". Runs unattended for hours — the user is not in the room.
argument-hint: "<canary spec number, e.g. 14>"
allowed-tools: "Bash(just *) Bash(git *) Bash(gh *) Bash(ssh comfy *) Bash(mutagen *) Bash(cd *) Agent"
---

Run the **`/strengthen-harness` skill** with canary spec `$ARGUMENTS`.

Follow the skill exactly. You don't implement anything yourself — you coordinate two subagents per loop (canary + fixer), review their code, decide merge-vs-close, and drive the loop. The user is not in the room; don't stop for anything short of a genuinely unresolvable block. Two consecutive buttery canary runs = terminate. 10 loops without convergence = escalate with diagnostics.

Merged PRs go to main. Closed PRs are expected (every canary is closed). Your first iteration starts immediately.
