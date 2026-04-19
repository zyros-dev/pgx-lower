#!/usr/bin/env python3
"""Baseline-aware wrapper around ctest for the pg_regress suite.

The pg_regress suite currently has many pre-existing failures that are not
caused by any one PR's change. Without a baseline, `just test` can never
actually gate anything — every PR inherits the red state. This wrapper
stores the known-failing set in `tests/pg_regress_baseline.txt` and
exits non-zero only on *delta*:

  - a test fails that used to pass (regression — blocking)
  - a test passes that used to fail (improvement — also reported, but
    surfaced so the author can remove it from the baseline)

Modes:
  --record      Run ctest, overwrite the baseline file with the current
                failure set, exit 0. Use when you've consciously accepted
                a new set of red tests on main.
  (default)     Run ctest, diff against baseline, exit 0 on match, 1 on
                new regressions.

Outputs a short delta summary; the raw ctest log is what stdin contained.

Usage inside the dev container:
    cd /workspace && ctest ... | python3 scripts/ptest_with_baseline.py

Or run ctest directly and pipe its output in — either way works. Reads
from argv's first positional for the baseline-file path override
(defaults to tests/pg_regress_baseline.txt relative to cwd).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Matches pg_regress TAP-ish lines, optionally prefixed by ctest's "N: " tag
# when run under `ctest -V` (verbose):
#   "ok 23        - 23_order_by_expressions                    79 ms"
#   "not ok 20    - 20_where_pattern_matching                 953 ms"
#   "1: ok 23     - 23_order_by_expressions                    79 ms"
LINE_RE = re.compile(r"^(?:\d+:\s+)?(not\s+ok|ok)\s+\d+\s*-\s*(\S+)")

# pg_regress emits "Bail out!" on stdout when it cannot even execute a
# test — most commonly because tests/sql/<NN>_name.sql exists but the
# matching tests/expected/<NN>_name.out file is missing (the RED half
# of a TDD pg_regress test). ctest itself exits non-zero in this case,
# but without detecting Bail out! here the baseline-delta summary would
# still print "OK: no new regressions vs baseline." — because no TAP
# `not ok` line is emitted for the missing-expected test. That produced
# two contradictory verdicts on the same transcript (non-zero exit AND
# "OK" tail). The Bail-out path now supersedes the baseline-OK branch
# with an explicit RED marker that names the missing test.
BAIL_OUT_RE = re.compile(r"(?:^|\s)Bail out!", re.MULTILINE)
# pg_regress prints "diff: /.../tests/expected/<NN>_name.out: No such
# file or directory" just before bailing. We recover the test name from
# that path so the RED marker can be concrete.
MISSING_EXPECTED_RE = re.compile(
    r"diff:\s+(?:\S*/)?tests/expected/([^/\s:]+?)\.out:\s*No such file or directory"
)


def parse_ctest_output(text: str) -> tuple[set[str], set[str]]:
    """Return (passing, failing) test-name sets from ctest/TAP-ish output."""
    passing: set[str] = set()
    failing: set[str] = set()
    for line in text.splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        status, name = m.group(1), m.group(2)
        if status == "ok":
            passing.add(name)
        else:
            failing.add(name)
    return passing, failing


def detect_bail_out(text: str) -> tuple[bool, list[str]]:
    """Return (bail_present, missing_expected_testnames).

    `missing_expected_testnames` is best-effort: we scan for the
    "diff: .../tests/expected/<name>.out: No such file" line pg_regress
    prints right before bailing. If Bail out! is present but no path is
    recoverable (different pg_regress version, truncated output), the
    caller still flags RED — just with a less specific marker.
    """
    if not BAIL_OUT_RE.search(text):
        return False, []
    missing = sorted(set(MISSING_EXPECTED_RE.findall(text)))
    return True, missing


def read_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text().splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def write_baseline(path: Path, failures: set[str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = header.rstrip() + "\n\n" + "\n".join(sorted(failures)) + "\n"
    path.write_text(body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-file", type=Path,
                    default=Path("tests/pg_regress_baseline.txt"),
                    help="path to the baseline failures file")
    ap.add_argument("--record", action="store_true",
                    help="overwrite baseline with current failures")
    args = ap.parse_args()

    raw = sys.stdin.read()
    passing, failing = parse_ctest_output(raw)
    bail_out, missing_expected = detect_bail_out(raw)

    if not passing and not failing and not bail_out:
        print("ERROR: no TAP-ish lines found in input — is ctest actually "
              "running with --output-on-failure?", file=sys.stderr)
        return 2

    if args.record:
        header = (
            "# pg_regress baseline: tests expected to fail on main.\n"
            "# Produced by `just test-record-baseline`. Review before committing.\n"
            "# A PR that adds a test name here must justify why (in the PR\n"
            "# description). A PR that removes a name — good, that test is\n"
            "# passing now — can do so freely."
        )
        write_baseline(args.baseline_file, failing, header)
        print(f"Recorded {len(failing)} failing tests to {args.baseline_file}")
        print(f"Passing: {len(passing)}, Failing: {len(failing)}")
        return 0

    baseline = read_baseline(args.baseline_file)

    new_failures = failing - baseline
    now_passing = baseline & passing   # previously-failing tests that passed this run
    still_failing = failing & baseline

    print("=== pg_regress delta vs baseline ===")
    print(f"Baseline known-failing: {len(baseline)}")
    print(f"This run:  passing={len(passing)}  failing={len(failing)}")
    print(f"  still failing (expected): {len(still_failing)}")
    print(f"  newly failing (regression): {len(new_failures)}")
    print(f"  newly passing (improvement): {len(now_passing)}")

    if now_passing:
        print("\nNEWLY PASSING — remove these from the baseline file:")
        for name in sorted(now_passing):
            print(f"  + {name}")

    # Bail out! supersedes the baseline-delta OK verdict. pg_regress
    # couldn't complete the suite, typically because a new .sql has no
    # matching .out — the classic TDD RED state. Print an unambiguous
    # single-verdict tail naming the missing test(s), then exit non-zero
    # regardless of whether the TAP `not ok` set matches the baseline.
    if bail_out:
        print("")
        if missing_expected:
            for name in missing_expected:
                print(f"NEW TEST NEEDS EXPECTED FILE: {name}")
            tests_list = ", ".join(missing_expected)
            print(
                f"\nRED: pg_regress bailed out — expected-output file(s) missing "
                f"for: {tests_list}."
            )
            print(
                "     Run `just expected-from-results <name>` once `just test` has "
                "produced the results/ output,\n"
                "     then re-run `just test` to confirm green. "
                "See .claude/skills/devops/SKILL.md step 2."
            )
        else:
            # Bail out! with no recoverable path — still RED, but we can't
            # name the test. Surface the raw marker so the agent at least
            # knows pg_regress short-circuited rather than passed.
            print("NEW TEST NEEDS EXPECTED FILE: <unknown — pg_regress "
                  "bailed without a 'diff: .../expected/<name>.out' line>")
            print("\nRED: pg_regress printed 'Bail out!' but the path could "
                  "not be parsed from the transcript. Inspect the full "
                  "ctest output above to identify the failing test.")
        return 1

    if new_failures:
        print("\nREGRESSIONS — these tests used to pass and now fail:")
        for name in sorted(new_failures):
            print(f"  ✘ {name}")
        print("\nFAIL: regressions detected. Fix the test(s) or, if the "
              "failure is genuinely expected, update the baseline with "
              "`just test-record-baseline` and justify in the PR body.")
        return 1

    print("\nOK: no new regressions vs baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
