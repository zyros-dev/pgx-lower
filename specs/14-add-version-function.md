# Spec 14 — Add `pgx_lower_version()` SQL function

**Tier:** smoke (workflow validation, not roadmap work)
**Stack on:** `main`
**Blocks:** —
**Estimated effort:** ~30 minutes of code, plus the harness round-trip

## Why this spec exists

This spec is a **deliberate smoke test of the agentic harness**, not a real
feature. Its job is to flush out bugs in the spec-claim / worktree / TDD /
build / bench / PR / review / merge / cleanup pipeline before we trust the
harness with any of the perf-roadmap specs (01–13).

If you're an agent reading this and wondering "why am I being asked to add
something so trivial?" — that's the point. Run the full devops loop, do
exactly what the spec says, no more.

## Goal

Expose the extension version as a SQL-callable function:

```sql
SELECT pgx_lower_version();   -- returns '1.0' (matches extension/control/pgx_lower.control)
```

## Background

The extension declares its version in `extension/control/pgx_lower.control`
(`default_version = '1.0'`). There is no SQL way to query this version at
runtime today. Adding one is cheap and exercises the full C-extension
boundary (PG_FUNCTION_INFO_V1, the install SQL script, regression test
harness) without touching anything performance-critical.

## What to change

### 1. C function

Add to `src/pgx-lower/execution/postgres/executor_c.c` near the existing
`pgx_lower_test_relalg` function (around line 107):

```c
PG_FUNCTION_INFO_V1(pgx_lower_version);
Datum pgx_lower_version(PG_FUNCTION_ARGS) {
    PG_RETURN_TEXT_P(cstring_to_text("1.0"));
}
```

The version string is hardcoded for now. (A future spec could pull it from
the control file or a build-time `#define`; out of scope here.)

### 2. SQL declaration

Add to `extension/sql/pgx_lower--1.0.sql`:

```sql
CREATE OR REPLACE FUNCTION pgx_lower_version()
RETURNS text
AS '$libdir/pgx_lower', 'pgx_lower_version'
LANGUAGE C IMMUTABLE STRICT;
```

### 3. Regression test (RED first — the test must fail before implementation)

Pick the next free test number — current registered tests in
`extension/CMakeLists.txt:39-85` go up to `42_*` plus the `tpch*` set.
Use **`16_version`** if free; otherwise the next available NN.

`tests/sql/16_version.sql`:
```sql
CREATE EXTENSION IF NOT EXISTS pgx_lower;
SELECT pgx_lower_version();
```

`tests/expected/16_version.out`:
```
CREATE EXTENSION
 pgx_lower_version
-------------------
 1.0
(1 row)

```

(Match the exact pg_regress output format — a trailing blank line, the
column-header gutter alignment, the row count line.)

### 4. Test registration

Add `16_version` to the `REGRESS` list in `extension/CMakeLists.txt`
(around line 39-85, in numeric order).

## Acceptance criteria

- `just check` clean.
- `just test` passes — including the new `16_version` regression test.
- All existing regression tests still pass (no collateral breakage).
- `just bench` produces a report; the chart should show **no meaningful
  movement** on any query (this spec doesn't touch any code path the TPC-H
  queries exercise). A YAY/NAY/MAYBE verdict of MAYBE (within the noise
  band) is the expected and correct outcome.
- `just bench-report` produces the `.db / .png / .md` triplet under
  `./benchmarks/`, all three committed with the PR.
- PR body includes the auto-generated benchmark report block. The Stats
  summary section will show roughly 0% deltas — that's correct for this
  spec.
- spec-reviewer subagent verdict: `approve` or `approve with comments`.
  If it returns `request changes` for substantive reasons, that's
  important data — surface it back to the user.

## Files to touch

| File | Change |
|------|--------|
| `src/pgx-lower/execution/postgres/executor_c.c` | Add `PG_FUNCTION_INFO_V1` + impl |
| `extension/sql/pgx_lower--1.0.sql` | Add `CREATE FUNCTION` |
| `tests/sql/16_version.sql` (new) | Two-line regression test |
| `tests/expected/16_version.out` (new) | Matching expected output |
| `extension/CMakeLists.txt` | Add `16_version` to `REGRESS` list |
| `./benchmarks/<prefix>.{db,png,md}` | Bench-report triplet from `just bench-report` |

## Risks

Effectively none. Worst-case the function returns the wrong string, in
which case the regression test fails and you fix the literal.

The reason this spec works as a smoke test is that the *risks are in the
harness, not the change*. If anything goes wrong, the failure is more
likely to be in `just spec-claim`, the mutagen sync, the build queue, or
the PR template than in the C code.

## A/B test

Standard `just bench` per the devops skill. Expected outcome: MAYBE
verdict (within noise band on every query). If you see a YAY or NAY
verdict, that's a harness bug — flag it; don't ship.

## Rollback

`git revert` of the merge commit. No data migration. The function and
SQL declaration disappear; existing tests keep passing.

## What this spec is NOT for

- Don't add a build-time version macro.
- Don't pull the version from the control file at runtime.
- Don't add other introspection functions while you're "in there."
- Don't refactor `executor_c.c` even if it looks tempting.

Stay scoped. The whole point is to test the harness, not to ship features.
The next agent will be running a real spec; leave the codebase looking
exactly like the codebase, plus one tiny new function.

## Reflection prompt (for after the PR lands)

When this spec is done, the user will ask the implementing agent something
like: **"Reflect on running spec 14 — what was hard, ambiguous, broken, or
slower than expected in the workflow?"** Capture honest answers; the
harness is what gets fixed, not the spec.
