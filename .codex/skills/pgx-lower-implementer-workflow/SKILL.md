---
name: pgx-lower-implementer-workflow
description: Use when asked to implement, finish, validate, or hand off a pgx-lower or pgx-cli wiki spec/plan, especially goals like "use implementer workflow on spec xyz".
---

# pgx-lower Implementer Workflow

## Short Goal

```text
/goal use pgx-lower-implementer-workflow on wiki/specs/pgx-cli/designs/2026-06-07-agent-implementation-efficiency-hardening-design.md
```

## Workflow

1. Read `AGENTS.md` first, then read the parent spec and every linked child plan before editing.
2. Record branch, status, prerequisites, intended commit boundary, dirty generated artifacts, ignored evidence paths, and any pre-existing failures.
3. If operating as a dev subagent under a coordinator, own the detailed implementation or spec patch and return a compact summary so the coordinator does not need to deep-dive the same work.
4. Use `pgx-cli dev preflight --strict` when available before expensive gates or thor work.
5. Make red tests before code for unit-testable behavior. Verify red failures prove intended behavior, not harness breakage.
6. Implement surgically. Run focused tests before broad gates, and reduce full-gate failures with the focused reproducer before rerunning the full gate.
7. Route high-risk commands through `pgx-cli`; use bounded `pgx-cli rg`, logs, and IR helpers instead of raw large-output reads.
8. Update `pgx-cli agent evidence` with claims, commands, run ids, transcript paths, and explicit deferrals.
9. Run an adversarial whole-branch review before claiming ready.
10. Use this reviewer prompt:

```text
Review the whole branch against the active spec and plan bundle. Validate that the implementation truthfully solves the problem, that all required gates and evidence are present, and that no important workflow path was skipped.
```

11. Fix valid reviewer findings. Counter invalid findings only with concrete source, spec, test, or gate evidence.
12. Accept QA findings from a coordinator, patch valid findings, and explicitly list any finding countered with source/spec/test evidence.
13. Run `pgx-cli pr ready` and the active plan's final gates before a ready claim.
14. Summarize run ids, transcript paths, evidence, PR state, dirty/generated artifacts, residual risks, and deferrals.

## Ready Claim

Only say the work is ready when acceptance claims map to evidence, required gates are fresh, reviewer/QA loops have no material unresolved findings, and any deferral is explicit.
