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


def load_pgx_times(db_path: Path) -> dict[str, float]:
    """Return {query_name: median duration_ms} for the most recent run's
    pgx_enabled=1 rows. A benchmark.db can hold many runs — only the latest
    by run_timestamp is used, so historical experiments don't skew the chart.
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
         WHERE q.pgx_enabled = 1
           AND r.run_id = (SELECT run_id FROM runs ORDER BY run_timestamp DESC LIMIT 1)
    """)
    buckets: dict[str, list[float]] = {}
    for qname, dur in cur.fetchall():
        if dur is None:
            continue
        buckets.setdefault(qname, []).append(float(dur))
    conn.close()
    return {q: median(v) for q, v in buckets.items() if v}


def load_meta(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT run_id, pgx_version, scale_factor, run_timestamp, label
          FROM runs ORDER BY run_timestamp DESC LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()
    if not row:
        return {}
    keys = ["run_id", "pgx_version", "scale_factor", "run_timestamp", "label"]
    return dict(zip(keys, row))


def compute_deltas(baseline: dict[str, float], current: dict[str, float]) -> list[tuple[str, float, float, float]]:
    """Return [(query, baseline_ms, current_ms, pct_improvement), ...] sorted by query."""
    shared = sorted(set(baseline) & set(current))
    out = []
    for q in shared:
        b = baseline[q]
        c = current[q]
        pct = (b - c) / b * 100.0 if b > 0 else 0.0
        out.append((q, b, c, pct))
    return out


def geomean_speedup(rows: list[tuple[str, float, float, float]]) -> float:
    """Geometric mean of baseline/current ratios, expressed as % improvement."""
    if not rows:
        return 0.0
    ratios = [b / c for (_, b, c, _) in rows if c > 0]
    if not ratios:
        return 0.0
    gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return (gm - 1.0) * 100.0


def plot_chart(rows, out_png: Path, title: str) -> None:
    queries = [q for (q, _, _, _) in rows]
    deltas = [d for (_, _, _, d) in rows]
    colors = [IMPROVE if d > 1 else REGRESS if d < -1 else NEUTRAL for d in deltas]

    fig, ax = plt.subplots(figsize=(max(8, len(queries) * 0.45), 5))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.bar(queries, deltas, color=colors, edgecolor="black", linewidth=0.6)

    for i, (q, _, _, d) in enumerate(rows):
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


def render_markdown(rows, baseline_meta, current_meta, chart_url: str) -> str:
    gm = geomean_speedup(rows)
    gm_label = f"{gm:+.1f}%" + (" 🟢" if gm > 1 else " 🔴" if gm < -1 else " ⚪")

    lines = []
    lines.append(f"## Benchmark report")
    lines.append("")
    lines.append(f"![bench diff]({chart_url})")
    lines.append("")
    lines.append(f"**Geometric-mean speedup vs baseline: {gm_label}**")
    lines.append("")
    lines.append(f"- Baseline: `{baseline_meta.get('pgx_version', '?')}` "
                 f"(run {baseline_meta.get('run_timestamp', '?')})")
    lines.append(f"- Current:  `{current_meta.get('pgx_version', '?')}` "
                 f"(run {current_meta.get('run_timestamp', '?')})")
    lines.append(f"- Scale factor: {current_meta.get('scale_factor', '?')}")
    lines.append("")
    lines.append("| Query | Baseline (ms) | Current (ms) | Δ |")
    lines.append("|-------|--------------:|-------------:|--:|")
    for q, b, c, d in rows:
        marker = "🟢" if d > 1 else "🔴" if d < -1 else "⚪"
        lines.append(f"| {q} | {b:.1f} | {c:.1f} | {d:+.1f}% {marker} |")
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
    rows = compute_deltas(baseline, current)
    if not rows:
        sys.exit("ERROR: no overlapping queries between baseline and current")

    png_path = args.out.with_suffix(".png")
    md_path = args.out.with_suffix(".md")

    plot_chart(rows, png_path, args.title)
    chart_url = args.chart_url or png_path.name
    md = render_markdown(rows, load_meta(args.baseline), load_meta(args.current), chart_url)
    md_path.write_text(md)

    print(md)
    print(f"\n[chart: {png_path}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
