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

Pick the **next available NN** — run `ls tests/sql/ | sort -n | tail -5`
to see the current high-water mark. Don't hard-code a number; the test
directory grows and slot suggestions drift. Substitute `<NN>` everywhere
below.

`tests/sql/<NN>_version.sql`:
```sql
CREATE EXTENSION IF NOT EXISTS pgx_lower;
SELECT pgx_lower_version();
```

**Do NOT write `tests/expected/<NN>_version.out` by hand.** pg_regress's
output format has footguns that almost always bite on the first attempt:

- Input SQL gets echoed before results.
- `NOTICE: ... already exists, skipping` fires because pg_regress
  pre-loads the extension, so `CREATE EXTENSION IF NOT EXISTS` is a
  no-op second load — the NOTICE IS part of the expected output.
- Column headers have trailing whitespace (`pgx_lower_version ` pads
  to the widest row), which editors strip on save.
- One blank line at the very end.

Generate the file from pg_regress directly. Run `just test` once (it
fails because the expected file doesn't exist, but writes the actual
result to `extension/results/<NN>_version.out`), then copy that back:

```sh
ssh comfy 'docker exec pgx-lower-dev cat /workspace/.worktrees/<slug>/extension/results/<NN>_version.out' \
  > tests/expected/<NN>_version.out
```

Or equivalently, `just expected-from-results <NN>_version` (same thing
wrapped).

This is the ONLY recommended path. The inline-described format above
(trailing space, NOTICE line, etc.) is for reference only — use it to
sanity-check what pg_regress wrote, not to construct the file yourself.

### 4. Test registration

Add `<NN>_version` to the `REGRESS` list in `extension/CMakeLists.txt`
(around line 39-85, in numeric order).

### 5. Unit test (mandatory — this spec exercises the utest path)

Spec 14 is a harness smoke test: **both** test paths (pg_regress and
gtest) need to be exercised so the harness itself is verified, not just
the code change. The regression test above covers pg_regress; add a
gtest for the function too.

Create `tests/unit/test_lowerings/test_version_function.cpp`:

```cpp
#include <gtest/gtest.h>
#include <string>

// pgx_lower_version returns a constant version string. The C wrapper
// hands it to PG via PG_RETURN_TEXT_P(cstring_to_text(...)), which
// requires a live PG backend. For the unit-test build we test the
// constant itself — trivial, but proves the utest path works
// end-to-end for this spec's pattern (function returns a literal string).
namespace {
constexpr const char* PGX_LOWER_VERSION = "1.0";
}

TEST(VersionFunctionTest, ReturnsOneDotZero) {
    EXPECT_STREQ(PGX_LOWER_VERSION, "1.0");
    EXPECT_EQ(std::string(PGX_LOWER_VERSION).size(), 3u);
}
```

Wire it into `tests/unit/test_lowerings/CMakeLists.txt` by mirroring an
existing small test block (`test_type_mapping` is the closest shape).
`just utest` must pass with this test registered.

If this feels like ceremony for a one-line constant return: it is, and
that's the point. Spec 14's whole job is exercising the harness, not
producing meaningful functionality. A future spec with real logic will
get a more substantive test using the same pattern.

## Acceptance criteria

- `just check-diff` clean (scoped to files this PR touches; `just check`
  whole-tree has pre-existing violations that aren't this PR's to fix).
- `just utest` passes — including the new `VersionFunctionTest`.
- `just test` passes — including the new `<NN>_version` regression test.
- All existing regression / unit tests still pass (no collateral breakage).
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
| `tests/sql/<NN>_version.sql` (new) | Two-line regression test |
| `tests/expected/<NN>_version.out` (new) | Expected output — **generated by pg_regress, not hand-written** |
| `extension/CMakeLists.txt` | Add `<NN>_version` to `REGRESS` list |
| `tests/unit/test_lowerings/test_version_function.cpp` (new) | Unit test — mandatory; exercises the utest path |
| `tests/unit/test_lowerings/CMakeLists.txt` | Register the new unit test (mirror `test_type_mapping` block) |
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
