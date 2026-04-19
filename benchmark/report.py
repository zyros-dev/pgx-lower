#!/usr/bin/env python3
"""PR benchmark report: compare pgx-lower timings on current branch vs a
main-branch baseline, emit a single bar chart (+% delta per TPC-H query)
and a markdown table suitable for a PR body.

Chart direction: bars point UP when the current branch is FASTER than
baseline (improvement), DOWN on regression. Flat means no change.

Writes:
  <out>.png  — bar chart
  <out>.md   — markdown report body

Exit code: 0 on success. Non-zero if data is missing or mismatched.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
IMPROVE = "#2ecc71"   # green
REGRESS = "#e74c3c"   # red
NEUTRAL = "#95a5a6"   # gray


def load_times(db_path: Path, pgx_enabled: int) -> dict[str, float]:
    """Return {query_name: median duration_ms} across iterations for the most
    recent run. Multiple iterations exist when --iterations > 1; taking the
    median damps JIT-compile variance that dominates at small scale factors.
    """
    if not db_path.exists():
        sys.exit(f"ERROR: db not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT q.query_name,
               json_extract(q.execution_metadata, '$.duration_ms') AS duration_ms
          FROM queries q
          JOIN runs   r ON q.run_id = r.run_id
         WHERE q.pgx_enabled = ?
           AND r.run_id = (SELECT run_id FROM runs ORDER BY run_timestamp DESC LIMIT 1)
    """, (pgx_enabled,))
    buckets: dict[str, list[float]] = {}
    for qname, dur in cur.fetchall():
        if dur is None:
            continue
        buckets.setdefault(qname, []).append(float(dur))
    conn.close()
    return {q: median(v) for q, v in buckets.items() if v}


def load_pgx_times(db_path: Path) -> dict[str, float]:
    return load_times(db_path, pgx_enabled=1)


def load_pg_times(db_path: Path) -> dict[str, float]:
    return load_times(db_path, pgx_enabled=0)


def load_validation(db_path: Path) -> dict[str, bool]:
    """Return {query_name: valid} for the most recent run's pgx_enabled=1 rows.

    `valid` is the per-query flag set by `validate_results()` in
    benchmark/tpch/run.py: it compares the output hash from pgx ON against
    the hash from pgx OFF and records whether they agree. A False here means
    pgx produced wrong output — a correctness regression, which must force
    the report verdict to NAY regardless of timing.
    """
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT q.query_name,
               json_extract(q.result_validation, '$.valid') AS valid
          FROM queries q
          JOIN runs   r ON q.run_id = r.run_id
         WHERE q.pgx_enabled = 1
           AND r.run_id = (SELECT run_id FROM runs ORDER BY run_timestamp DESC LIMIT 1)
    """)
    out: dict[str, bool] = {}
    for qname, valid in cur.fetchall():
        if valid is None:
            continue
        # sqlite stores booleans as 0/1 ints via json_extract.
        out[qname] = bool(valid)
    conn.close()
    return out


def load_meta(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT run_id, pgx_version, scale_factor, iterations, run_timestamp, label
          FROM runs ORDER BY run_timestamp DESC LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()
    if not row:
        return {}
    keys = ["run_id", "pgx_version", "scale_factor", "iterations", "run_timestamp", "label"]
    return dict(zip(keys, row))


def compute_deltas(
    baseline: dict[str, float],
    current: dict[str, float],
    pg: dict[str, float],
) -> list[tuple[str, float, float, float, float]]:
    """Return [(query, pg_ms, baseline_ms, current_ms, pct_improvement), ...].

    pct_improvement = (baseline - current) / baseline * 100, so positive means
    this PR is faster than the main baseline. pg_ms is the PostgreSQL time
    from the current run (not the baseline) for the "how close is pgx to PG"
    context line in the table.
    """
    shared = sorted(set(baseline) & set(current))
    out = []
    for q in shared:
        b = baseline[q]
        c = current[q]
        pct = (b - c) / b * 100.0 if b > 0 else 0.0
        out.append((q, pg.get(q, float("nan")), b, c, pct))
    return out


def geomean_speedup(rows) -> float:
    """Geometric mean of baseline/current ratios, expressed as % improvement."""
    if not rows:
        return 0.0
    ratios = [b / c for (_, _, b, c, _) in rows if c > 0]
    if not ratios:
        return 0.0
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return (gm - 1.0) * 100.0


def verdict(rows, invalid_queries: list[str] | None = None) -> tuple[str, str]:
    """Return (tag, label) for the PR verdict.

    Correctness dominates: if any query produced wrong output (pgx hash
    didn't match PG hash), the verdict is NAY no matter what the timing
    numbers say. A fast wrong answer is still wrong.

    Timing thresholds (calibrated for SF=0.5 iter=1, where execution
    dominates JIT compile and per-query variance is ~5-10%):

      YAY   — geomean ≥ +3% AND no query regresses worse than −5%.
      NAY   — geomean ≤ −3% OR any query regresses worse than −10%.
      MAYBE — everything in between (inside the noise band, mixed signals).

    An earlier version had JIT-compile-noise softening to turn single-query
    outliers at SF=0.01 into MAYBE instead of NAY. That was a workaround
    for SF=0.01 iter=5 being bimodal on q01. Now that the bench runs at
    SF=0.5 (execution-dominated), the softening is both unneeded and
    actively wrong — a real -10% regression on a high-pgx-time query
    would be mislabeled as compile noise. Removed.
    """
    if invalid_queries:
        return ("nay", f"🔴 NAY — correctness regression ({', '.join(invalid_queries)})")
    gm = geomean_speedup(rows)
    worst = min((d for (_, _, _, _, d) in rows), default=0.0)
    if gm <= -3.0 or worst <= -10.0:
        return ("nay", "🔴 NAY — regression")
    if gm >= 3.0 and worst > -5.0:
        return ("yay", "🟢 YAY — improvement")
    return ("maybe", "🟡 MAYBE — in the noise band")


def plot_chart(rows, out_png: Path, title: str) -> None:
    queries = [q for (q, _, _, _, _) in rows]
    deltas = [d for (_, _, _, _, d) in rows]
    colors = [IMPROVE if d > 1 else REGRESS if d < -1 else NEUTRAL for d in deltas]

    fig, ax = plt.subplots(figsize=(max(8, len(queries) * 0.45), 5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(queries)))
    ax.bar(queries, deltas, color=colors, edgecolor="black", linewidth=0.6)

    for i, (_, _, _, _, d) in enumerate(rows):
        y = d + (1 if d >= 0 else -1) * max(abs(max(deltas, default=1)), 1) * 0.03
        ax.text(i, y, f"{d:+.1f}%", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=8)

    ax.set_ylabel("pgx-lower speedup vs baseline (%)", fontweight="bold")
    ax.set_xlabel("TPC-H query")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticklabels(queries, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_markdown(rows, baseline_meta, current_meta, chart_url: str,
                    validation: dict[str, bool] | None = None) -> str:
    validation = validation or {}
    invalid_queries = sorted(q for q, v in validation.items() if not v)
    gm = geomean_speedup(rows)
    gm_label = f"{gm:+.1f}%" + (" 🟢" if gm > 1 else " 🔴" if gm < -1 else " ⚪")
    _, verdict_label = verdict(rows, invalid_queries=invalid_queries)

    lines = []
    lines.append(f"## {verdict_label}")
    lines.append("")
    if invalid_queries:
        lines.append(f"> **Correctness regression:** pgx output hash did not "
                     f"match PostgreSQL for: `{', '.join(invalid_queries)}`. "
                     f"Timing numbers below are reported for context but "
                     f"don't change the verdict — a fast wrong answer is "
                     f"still wrong.")
        lines.append("")
    lines.append(f"- **Geomean speedup vs baseline:** {gm_label}")
    lines.append(f"- **Scale factor:** {current_meta.get('scale_factor', '?')}, "
                 f"iterations: {current_meta.get('iterations', '?')}")
    lines.append(f"- **Baseline:** `{baseline_meta.get('pgx_version', '?')}` "
                 f"({baseline_meta.get('run_timestamp', '?')})")
    lines.append(f"- **Current:**  `{current_meta.get('pgx_version', '?')}` "
                 f"({current_meta.get('run_timestamp', '?')})")
    lines.append("")
    lines.append(f"![bench diff]({chart_url})")
    lines.append("")
    lines.append("PG column is PostgreSQL-only runtime from the current run — "
                 "reference for absolute pgx-lower cost, not part of the A/B.")
    lines.append("")
    lines.append("| Query | PG (ms) | pgx baseline (ms) | pgx current (ms) | Δ | ✓ |")
    lines.append("|-------|--------:|------------------:|-----------------:|--:|:--|")
    for q, pg, b, c, d in rows:
        marker = "🟢" if d > 1 else "🔴" if d < -1 else "⚪"
        pg_str = f"{pg:.1f}" if pg == pg else "—"
        if q not in validation:
            check = "—"
        elif validation[q]:
            check = "✅"
        else:
            check = "❌"
        lines.append(f"| {q} | {pg_str} | {b:.1f} | {c:.1f} | {d:+.1f}% {marker} | {check} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True, help="baseline benchmark.db")
    ap.add_argument("--current", type=Path, required=True, help="current-branch benchmark.db")
    ap.add_argument("--out", type=Path, required=True, help="output prefix (creates .png and .md)")
    ap.add_argument("--chart-url", type=str, default=None,
                    help="URL to embed for the chart in the markdown body "
                         "(defaults to relative .png path)")
    ap.add_argument("--title", type=str, default="pgx-lower: this PR vs main baseline")
    args = ap.parse_args()

    baseline = load_pgx_times(args.baseline)
    current = load_pgx_times(args.current)
    pg = load_pg_times(args.current)
    validation = load_validation(args.current)
    rows = compute_deltas(baseline, current, pg)
    if not rows:
        sys.exit("ERROR: no overlapping queries between baseline and current")

    png_path = args.out.with_suffix(".png")
    md_path = args.out.with_suffix(".md")

    plot_chart(rows, png_path, args.title)
    chart_url = args.chart_url or png_path.name
    md = render_markdown(rows, load_meta(args.baseline), load_meta(args.current),
                         chart_url, validation=validation)
    md_path.write_text(md)

    print(md)
    print(f"\n[chart: {png_path}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
