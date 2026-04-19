#!/usr/bin/env python3
"""Unit tests for scripts/ptest_with_baseline.py.

Covers the "Bail out!" RED-phase detection added in harness round 9: when
a new tests/sql/<NN>_name.sql exists without a matching
tests/expected/<NN>_name.out, pg_regress prints "Bail out!" and the
baseline-delta summary used to print a misleading "OK: no new regressions"
verdict. The script must now surface an unambiguous RED marker and exit
non-zero when Bail out! is present, regardless of baseline delta.

Run:
    python3 scripts/test_ptest_with_baseline.py
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).with_name("ptest_with_baseline.py")


def _run(stdin_text: str, baseline_text: str = "") -> subprocess.CompletedProcess:
    """Run ptest_with_baseline.py with stdin + a tempfile baseline.

    Returns the completed process (stdout/stderr captured, exit code preserved).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(baseline_text)
        baseline_path = f.name
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline-file", baseline_path],
        input=stdin_text,
        capture_output=True,
        text=True,
    )


# ---------- existing-behavior regression tests --------------------------------

def test_all_passing_no_baseline_exits_zero():
    stdin = textwrap.dedent("""\
        1: ok 1 - 1_one_tuple          10 ms
        1: ok 2 - 2_two_tuples         11 ms
    """)
    r = _run(stdin, baseline_text="")
    assert r.returncode == 0, r.stderr
    assert "OK: no new regressions vs baseline." in r.stdout
    assert "NEW TEST NEEDS EXPECTED FILE" not in r.stdout
    print("PASS: all passing, no baseline -> exit 0, no RED marker")


def test_known_failure_in_baseline_exits_zero():
    stdin = textwrap.dedent("""\
        1: ok 1 - 1_one_tuple          10 ms
        1: not ok 2 - 2_known_red      50 ms
    """)
    r = _run(stdin, baseline_text="2_known_red\n")
    assert r.returncode == 0, r.stderr
    assert "OK: no new regressions vs baseline." in r.stdout
    print("PASS: known baseline failure -> exit 0")


def test_new_failure_exits_one():
    stdin = textwrap.dedent("""\
        1: ok 1 - 1_one_tuple          10 ms
        1: not ok 2 - 2_new_regression 50 ms
    """)
    r = _run(stdin, baseline_text="")
    assert r.returncode == 1
    assert "REGRESSIONS" in r.stdout
    print("PASS: new failure not in baseline -> exit 1")


# ---------- NEW: Bail out! RED-phase tests ------------------------------------

def test_bail_out_emits_explicit_red_marker():
    """pg_regress prints 'Bail out!' when a new .sql has no .out file.

    Before the fix: the baseline script only looked at ok/not-ok TAP lines
    and would report "OK: no new regressions vs baseline." even though
    ctest itself exited non-zero with Bail out! — two contradictory
    verdicts for the same transcript.

    After the fix: when Bail out! is present, the script emits a clear
    RED marker line identifying which test(s) need expected files, exits
    non-zero, and does NOT print the misleading "OK: no new regressions"
    tail.
    """
    # Realistic slice of what `ctest -V` emits when tests/sql/43_version.sql
    # exists but tests/expected/43_version.out does not. pg_regress runs
    # the prior tests, then bails when it hits the missing expected file.
    stdin = textwrap.dedent("""\
        1: ok 1 - 1_one_tuple          10 ms
        1: ok 2 - 2_two_tuples         11 ms
        1: # initializing database system
        1: # using temp instance on port 57636
        1: diff: /workspace/tests/expected/43_version.out: No such file or directory
        1: diff command failed with status 512: diff "/workspace/tests/expected/43_version.out" "/workspace/.../results/43_version.out" > "/workspace/.../results/43_version.out.diff" 2>&1
        1: Bail out!diff command failed
    """)
    r = _run(stdin, baseline_text="")
    assert r.returncode != 0, (
        "Bail out! in stdin must make the script exit non-zero, even when "
        f"no 'not ok' lines match the baseline. stdout was:\n{r.stdout}"
    )
    # The explicit RED marker must name the missing-expected test so the
    # agent reading the transcript knows which file to author next.
    assert "NEW TEST NEEDS EXPECTED FILE" in r.stdout, (
        f"expected RED marker in stdout, got:\n{r.stdout}"
    )
    assert "43_version" in r.stdout, (
        f"RED marker should name the test (43_version); got:\n{r.stdout}"
    )
    # Must NOT print the misleading "OK: no new regressions" tail.
    assert "OK: no new regressions vs baseline." not in r.stdout, (
        f"bail-out run must not print the baseline-OK tail; got:\n{r.stdout}"
    )
    print("PASS: Bail out! -> explicit RED marker + non-zero exit + no OK tail")


def test_bail_out_supersedes_baseline_ok():
    """Even when there are NO baseline-delta regressions in the TAP lines,
    a Bail out! must still trip the RED state. The baseline script's
    "everything's fine per the delta" view is superseded by the fact that
    pg_regress couldn't complete the suite."""
    stdin = textwrap.dedent("""\
        1: ok 1 - 1_one_tuple          10 ms
        1: ok 2 - 2_two_tuples         11 ms
        1: diff: /workspace/tests/expected/43_version.out: No such file or directory
        1: Bail out!diff command failed
    """)
    r = _run(stdin, baseline_text="")
    assert r.returncode != 0
    # The script should not claim "OK" when Bail out! is in the transcript.
    assert "NEW TEST NEEDS EXPECTED FILE" in r.stdout
    print("PASS: Bail out! supersedes baseline-delta OK")


def test_bail_out_without_path_still_red():
    """Bail out! might appear without a parseable 'diff: <path>' line
    (different pg_regress versions, truncated output, etc.). Even in that
    degenerate case, the script must still flag RED — just with a less
    specific marker."""
    stdin = "1: ok 1 - 1_one_tuple 10 ms\n1: Bail out!\n"
    r = _run(stdin, baseline_text="")
    assert r.returncode != 0
    assert "NEW TEST NEEDS EXPECTED FILE" in r.stdout or "BAIL OUT" in r.stdout
    print("PASS: bare Bail out! still flags RED")


if __name__ == "__main__":
    tests = [
        test_all_passing_no_baseline_exits_zero,
        test_known_failure_in_baseline_exits_zero,
        test_new_failure_exits_one,
        test_bail_out_emits_explicit_red_marker,
        test_bail_out_supersedes_baseline_ok,
        test_bail_out_without_path_still_red,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
    if failed:
        print(f"\n{failed} test(s) failed out of {len(tests)}")
        raise SystemExit(1)
    print(f"\nAll {len(tests)} tests passed.")
