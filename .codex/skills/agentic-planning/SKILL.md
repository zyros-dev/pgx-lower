---
name: agentic-planning
description: Use when writing, reviewing, approving, or handing off implementation plans for agentic coding work, especially multi-file or multi-agent plans where tests, verification, edge cases, tooling choices, wiki/spec updates, or execution order must be checked before dispatch.
---

# Agentic Planning

## Purpose

Use this as a companion to planning and execution skills. It does not replace
`superpowers:writing-plans`, `superpowers:executing-plans`, or project-specific
workflows. It is a checklist for making plans safe and useful for agents.

Good plans ground truth. They state what must become true, how the agent proves
that truth, and how another agent can resume without reinterpreting chat history.

Repo-local discoverability: question discipline, reviewer preemption, and
spec-vs-plan timing are mandatory plan-audit areas.

## Truth Claims

Start by extracting the plan's behavioral truth claims:

- What behavior, artifact, route, output, config, error, or invariant must become true?
- What existing behavior must remain true?
- What is explicitly out of scope?
- Which claims are structural only, and which are behavioral?

Rewrite vague goals until each claim can be checked. If a claim cannot be
checked automatically or manually with concrete evidence, mark it as deferred or
review policy instead of pretending it is verified.

## Red-Green Grounding

Prefer red-green tests for every unit-testable truth claim.

Red-green testing is truth grounding for agents: write the expected behavior as
a failing test, observe that it fails for the right reason, then implement until
it passes. This validates the agent's understanding before it changes code.

For each unit-testable claim, the plan should say:

- the test file to create or modify;
- the exact failing test behavior;
- the command to run in the red state;
- the expected red failure;
- the minimal implementation target;
- the green command and expected pass.

Use integration/regression tests after unit anchors to prove composition, not as
a substitute for small local truth checks.

## Question Discipline

Ask fewer, higher-leverage questions.

When a principle is already decided, stop asking the same question in scenario
form. If the user has chosen "PostgreSQL truthfulness," do not repeatedly ask
whether each type-system case should respect PostgreSQL. State the principle,
list the consequences, and ask only about real tradeoffs such as scope,
testability, sequencing, or fallback behavior.

Ask 2-3 questions at a time when questions are genuinely needed. Prefer
answering routine placement questions yourself by inspecting the codebase.

## Verification Alignment

Every verification step must match the claim it is supposed to prove.

For each verification, name:

- **Claim:** the exact truth being checked.
- **Method:** structural, automated behavioral, or manual observation.
- **Action:** command to run or artifact to inspect.
- **Expected evidence:** output, failure, file, notice, summary, state, or diff.

Structural checks include builds, typechecks, and lint. They prove structure only:
the code compiles, links, or follows static rules under that configuration.

Behavioral checks include unit tests, regression tests, CLI invocations, SQL
queries, runtime assertions, and golden-output checks. They must directly observe
the claimed behavior.

Manual checks are acceptable only when the plan states exactly what to inspect
and what evidence to record. "Inspect manually" is not enough.

Reject verification like "run the build" for a behavioral claim unless there is
also a behavioral or manual observation that directly tests the claim.

Verification must prove the specific claim. "It exists" and "it compiles" are
not enough for a workflow, route, output, or runtime behavior unless
existence/compilation is the actual claim. If a plan needs manual testing, say
exactly how the agent tests it and what observation counts as success.

## Edge Cases

Make edge cases a first-class section for each component:

- normal path;
- missing, null, empty, invalid, malformed input;
- duplicate IDs, keys, files, events, or outputs;
- disabled, forced, fallback, or bypass modes;
- order changes and repeated events;
- unrelated output that must be preserved;
- smallest case, largest reasonable case, and representative real case;
- existing behavior that must not change.

Each important edge case should have a unit test, integration test, manual check,
or explicit deferred-risk note.

Include edge cases early, not as cleanup. Common misses include missing IDs,
invalid syntax, stale generated artifacts, fallback/force modes, route noise,
null/empty values, duplicate entries, and broad commands that produce too much
output.

## Tooling Discipline

Prefer the project's existing tool entrypoint. If a repo has a typed CLI such as
`pgx-cli`, reusable agent/dev workflows should usually live there.

Before adding helper scripts, ask:

- Is this reusable enough to belong in the project CLI?
- Does it need strict types or unit tests?
- Will agents discover it naturally?
- Does it create a second way to do the same thing?
- Who owns or removes it after the plan?

Do not scatter random shell or Python scripts around the codebase unless the plan
justifies why they cannot live in the project toolkit. Temporary tooling must
have a cleanup point.

Avoid casual custom parsers, linters, or static checkers. If a plan introduces
one, require a clear reason, tests, and ownership.

For pgx-lower, prefer TypeScript inside `pgx-cli` for reusable agent/dev
tooling. Do not scatter random shell or Python scripts around the repo unless
the plan explicitly justifies the exception and includes cleanup.

## Spec vs Plan Timing

Specs may be queued ahead of implementation. Plans can also be queued ahead, but
they must be re-audited immediately before dispatch because code, tooling, and
agent behavior may have changed.

Before dispatch:

- reread the parent spec and every linked child plan;
- check paths and command names against the current repo;
- remove stale assumptions;
- ensure verification still proves the intended behavior;
- record any approved exception to normal gates or PR boundaries.

Do not treat an old plan as ready merely because it was once reviewed.

## Handoff Safety

Before dispatching an implementer:

- Open and review every plan file in the bundle, not just the parent spec.
- List the parent spec, plan order, prerequisites, and known caveats.
- Name generated files, ignored files, and committed files separately.
- State commit boundaries and verification commands per checkpoint.
- Require staging only intended files and preserving unrelated dirty worktree
  changes.
- Keep durable decisions in the wiki/specs and commit them.
- Check cross-plan consistency for names, formats, config keys, paths, route
  markers, commands, and expected output.
- Decide whether child plans are independently gate-green vertical slices or an
  approved horizontal/stacked exception. Do not imply both.

Plans should be resumable. A later agent should be able to continue from any
checkpoint using the written plan and repository state, without needing hidden
conversation context.

## Reviewer Preemption

Do not leave known mistakes for a reviewer to find.

Before saying a plan is ready, do your own adversarial pass. If you can already
predict a reviewer finding, fix it. Reviewers should find things you missed, not
things you knowingly left behind.

When receiving review feedback, distinguish real contract gaps from preferences.
Patch real gaps. Counter questionable feedback only with concrete source, test,
or spec evidence.

## Plan Audit

Before approving a plan, scan for:

- prose-only behavior with no test or observation;
- verification that proves a nearby fact instead of the stated claim;
- missing red tests for unit-testable behavior;
- no edge-case coverage;
- broad "implement feature" steps without small checkpoints;
- new scripts or tools outside the project toolkit;
- unowned generated files or expected-output churn;
- unclear rollback, cleanup, or commit boundaries;
- specs/wiki decisions that were discussed but not recorded;
- inconsistencies across sibling plan files;
- known reviewer-obvious mistakes that have not been fixed.
