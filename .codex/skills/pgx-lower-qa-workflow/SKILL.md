---
name: pgx-lower-qa-workflow
description: Use when asked to review, QA, validate, or adversarially inspect a pgx-lower or pgx-cli branch against a wiki spec or plan bundle.
---

# pgx-lower QA Workflow

## Short Goal

```text
/goal use pgx-lower-qa-workflow on feat/some-branch against wiki/specs/pgx-cli/designs/2026-06-07-agent-implementation-efficiency-hardening-design.md
```

## QA Order

Always read the parent spec and read every linked child plan before code review.

1. Read the parent spec.
2. Read every linked child plan.
3. If operating as a QA subagent under a coordinator, own the adversarial detail pass and do not assume the coordinator has personally audited every spec or code path.
4. List acceptance criteria, verification commands, required gates, and evidence artifacts before reading implementation details.
5. Run or validate the stated gates where available. If a gate is unavailable, report why and what evidence remains missing.
6. Inspect run artifacts, summaries, transcripts, evidence files, generated artifacts, and readiness state for freshness and relevance.
7. Read the code after understanding the contract.
8. Check tests prove claimed behavior, not only compilation, fixtures, or harness behavior.
9. Check raw-output bypasses, stale-state bypasses, generated-artifact bypasses, and readiness bypasses.
10. Report findings first, ordered by severity, with file/line references and the violated spec, plan, test, or readiness contract.
11. Distinguish real contract gaps from preferences so the dev subagent can either fix the issue or counter it with evidence in the next loop.
12. Explicitly say when no issue is found and what residual risk remains.

## Severity

- `P0`: blocks correctness, safety, or required readiness gates.
- `P1`: violates acceptance criteria, required workflow, or evidence integrity.
- `P2`: meaningful maintainability, coverage, or handoff risk.
- `P3`: preference or cleanup; do not frame as a blocking finding.
