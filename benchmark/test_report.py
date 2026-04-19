#!/usr/bin/env python3
"""Unit tests for benchmark/report.py's verdict logic.

Focused on the 'compile-noise outlier' softening path — the spec-14
reviewer reported a 🔴 NAY on a change that touched zero execution-path
code, triggered by q01 randomly landing at -54% under JIT-compile
variance at SF=0.01. The softening turns that specific pattern into a
🟡 MAYBE with a call-to-action instead of a hard stop.

Run: python3 benchmark/test_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from report import verdict, _compile_noise_outlier


def row(query, pg_ms, baseline_ms, current_ms):
    """Build a row tuple matching compute_deltas' output shape."""
    pct = (baseline_ms - current_ms) / baseline_ms * 100.0 if baseline_ms > 0 else 0.0
    return (query, pg_ms, baseline_ms, current_ms, pct)


def test_spec14_outlier_is_softened_to_maybe():
    """The exact failure the spec-14 reviewer hit: q01 -54%, 19 other
    queries ranging +1% to +16%, geomean +1.4%. Should downgrade from
    NAY to MAYBE. A 20-query payload matches the actual SF=0.01 run.
    """
    # q01: the noisy compile-bound outlier (367ms → 565ms = -54%).
    rows = [row("q01", 60, 367, 565)]
    # 19 other queries each at a uniform +5% speedup (current = baseline * 0.95).
    # Mathematically: one -54% ratio plus 19 × +5% ratios averages to ~+2.5%
    # geomean — the 'positive-but-for-q01' shape the reviewer observed.
    for i in range(2, 23):
        if i in (17, 20):
            continue  # skipped in bench recipe; mirror that
        baseline_ms = 200 + i * 15
        current_ms = baseline_ms * 0.95  # each +5%
        rows.append(row(f"q{i:02d}", 10 + i, baseline_ms, current_ms))

    gm = __import__("report").geomean_speedup(rows)
    assert gm > 0, f"test setup wrong — geomean should be positive, got {gm}"

    tag, label = verdict(rows)
    assert tag == "maybe", f"expected maybe, got {tag}: {label}"
    assert "compile-noise" in label.lower(), f"label doesn't explain why: {label}"
    assert "q01" in label, f"label doesn't name the outlier: {label}"
    assert "bench-merge" in label, f"label doesn't suggest next step: {label}"
    print(f"PASS: spec14 outlier softened to MAYBE (geomean={gm:+.1f}%)")


def test_real_regression_still_triggers_nay():
    """A broad-based regression (not just one outlier) should still NAY."""
    rows = [
        row("q01", 60, 300, 400),   # -33%
        row("q02", 8, 500, 600),    # -20%
        row("q03", 11, 400, 450),   # -13%
        row("q04", 4, 200, 220),    # -10%
        row("q05", 37, 300, 310),   # -3%
    ]
    tag, label = verdict(rows)
    assert tag == "nay", f"expected nay, got {tag}: {label}"
    assert "compile-noise" not in label.lower(), f"false-positive soften: {label}"
    print("PASS: broad regression still NAY")


def test_small_query_outlier_still_triggers_nay():
    """A -54% regression on a sub-100ms pgx query is suspicious (exec-time
    matters at that scale, not just compile) — keep NAY, don't soften."""
    rows = [
        row("q01", 5, 50, 77),      # -54% BUT only 77ms → likely real
        row("q02", 8, 500, 500),    # 0%
        row("q03", 11, 400, 400),   # 0%
    ]
    tag, label = verdict(rows)
    assert tag == "nay", f"expected nay, got {tag}: {label}"
    print("PASS: sub-100ms outlier still NAY")


def test_correctness_failure_always_nay():
    """Invalid queries override everything, even a perfect geomean."""
    rows = [
        row("q01", 60, 300, 250),  # +17% (great)
        row("q02", 8, 500, 450),   # +10%
    ]
    tag, label = verdict(rows, invalid_queries=["q01"])
    assert tag == "nay", f"expected nay, got {tag}"
    assert "correctness" in label.lower()
    assert "q01" in label
    print("PASS: correctness failure overrides timing")


def test_two_outliers_still_nay():
    """Softening only applies when there's exactly one outlier — two
    outliers means something systematic is wrong, keep NAY."""
    rows = [
        row("q01", 60, 400, 600),  # -50%
        row("q02", 60, 400, 600),  # -50%
        row("q03", 11, 400, 390),  # +2.5%
        row("q04", 11, 400, 390),  # +2.5%
    ]
    tag, label = verdict(rows)
    assert tag == "nay", f"expected nay, got {tag}: {label}"
    print("PASS: two outliers still NAY")


def test_yay_passes_through():
    """Clean uniform improvement should YAY."""
    rows = [
        row("q01", 60, 400, 350),  # +12.5%
        row("q02", 8, 500, 440),   # +12%
        row("q03", 11, 400, 360),  # +10%
    ]
    tag, label = verdict(rows)
    assert tag == "yay", f"expected yay, got {tag}"
    print("PASS: clean improvement YAY")


def test_outlier_detector_unit():
    """Direct tests on _compile_noise_outlier — the threshold edges."""
    # Geomean negative → not softened even if one outlier
    rows = [
        row("q01", 60, 400, 600),  # -50%
        row("q02", 8, 500, 600),   # -20%
    ]
    assert _compile_noise_outlier(rows) is None, "gm negative shouldn't soften"

    # Clean case — geomean strongly positive, one bad
    rows = [
        row("q01", 60, 400, 600),
        row("q02", 8, 500, 400),
        row("q03", 11, 400, 320),
    ]
    out = _compile_noise_outlier(rows)
    assert out is not None, "should detect softening case"
    assert out[0] == "q01"

    # Outlier too extreme (-70%) → real regression, not noise
    rows = [
        row("q01", 60, 400, 700),  # -75%
        row("q02", 8, 500, 400),
        row("q03", 11, 400, 320),
    ]
    assert _compile_noise_outlier(rows) is None, "-75% is too extreme for compile noise"
    print("PASS: outlier detector threshold edges")


if __name__ == "__main__":
    test_spec14_outlier_is_softened_to_maybe()
    test_real_regression_still_triggers_nay()
    test_small_query_outlier_still_triggers_nay()
    test_correctness_failure_always_nay()
    test_two_outliers_still_nay()
    test_yay_passes_through()
    test_outlier_detector_unit()
    print("\nAll tests passed.")
