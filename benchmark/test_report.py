#!/usr/bin/env python3
"""Unit tests for benchmark/report.py's verdict logic.

Run: python3 benchmark/test_report.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from report import verdict


def row(query, pg_ms, baseline_ms, current_ms):
    """Build a row tuple matching compute_deltas' output shape."""
    pct = (baseline_ms - current_ms) / baseline_ms * 100.0 if baseline_ms > 0 else 0.0
    return (query, pg_ms, baseline_ms, current_ms, pct)


def test_clean_improvement_is_yay():
    rows = [
        row("q01", 60, 400, 350),   # +12.5%
        row("q02", 8, 500, 440),    # +12%
        row("q03", 11, 400, 360),   # +10%
    ]
    tag, _ = verdict(rows)
    assert tag == "yay"
    print("PASS: clean improvement YAY")


def test_broad_regression_is_nay():
    """Geomean dragged below -3% — real slowdown, NAY."""
    rows = [
        row("q01", 60, 300, 400),   # -33%
        row("q02", 8, 500, 600),    # -20%
        row("q03", 11, 400, 450),   # -13%
        row("q04", 4, 200, 220),    # -10%
        row("q05", 37, 300, 310),   # -3%
    ]
    tag, _ = verdict(rows)
    assert tag == "nay"
    print("PASS: broad regression NAY")


def test_single_query_regression_past_10pct_is_nay():
    """At SF=0.5 iter=1, execution dominates. A -15% on any single query
    is a real signal, not JIT noise — gate must fire.

    Earlier versions had compile-noise-outlier softening that would have
    MAYBE'd this; removed when moving off the SF=0.01 bimodal regime.
    """
    rows = [
        row("q01", 100, 5000, 5800),   # -16%
        row("q02", 50, 2000, 2020),    # -1%
        row("q03", 80, 3000, 3010),    # -0.3%
    ]
    tag, _ = verdict(rows)
    assert tag == "nay"
    print("PASS: single-query -16% is NAY (no softening at SF=0.5)")


def test_noise_band_is_maybe():
    """Geomean and per-query deltas inside ±3% — noise-band run."""
    rows = [
        row("q01", 60, 400, 405),   # -1.2%
        row("q02", 8, 500, 498),    # +0.4%
        row("q03", 11, 400, 395),   # +1.3%
    ]
    tag, _ = verdict(rows)
    assert tag == "maybe"
    print("PASS: noise-band MAYBE")


def test_correctness_failure_always_nay():
    """Invalid queries override timing, even a perfect geomean."""
    rows = [
        row("q01", 60, 300, 250),   # +17%
        row("q02", 8, 500, 450),    # +10%
    ]
    tag, label = verdict(rows, invalid_queries=["q01"])
    assert tag == "nay"
    assert "correctness" in label.lower()
    assert "q01" in label
    print("PASS: correctness failure overrides timing")


def test_borderline_yay_just_below_threshold():
    """Geomean +2.8% (just below YAY threshold) → MAYBE."""
    rows = [
        row("q01", 60, 400, 389),  # +2.75%
        row("q02", 8, 500, 486),   # +2.8%
    ]
    tag, _ = verdict(rows)
    assert tag == "maybe"
    print("PASS: geomean +2.8% is MAYBE (just below YAY)")


if __name__ == "__main__":
    test_clean_improvement_is_yay()
    test_broad_regression_is_nay()
    test_single_query_regression_past_10pct_is_nay()
    test_noise_band_is_maybe()
    test_correctness_failure_always_nay()
    test_borderline_yay_just_below_threshold()
    print("\nAll tests passed.")
