"""F2. Stage 1 prefilter campaign R@200 per lever.

Data source:
    part/reports/prefilter_campaign_20260419_results.md §single-lever-deltas
    (BASELINE 0.9166, E1 edge 0.9181, E2a edge+k16+r3 0.9227,
     E2c edge+k32+r3 0.9223, E6 edge+PNA 0.9150, E8 r=0 0.8641)

Rendered to:
    LaTex/figures/chapter_4/prefilter_levers.pdf
    LaTex/figures/chapter_4/prefilter_levers.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Source: prefilter_campaign_20260419_results.md results table
LEVERS = [
    ("E8 (no aggregation)", 0.8641, "#555555"),
    ("Baseline", 0.9166, "#6c757d"),
    ("E6 (PNA)", 0.9150, "#7f8fa6"),
    ("E1 (edge only)", 0.9181, "#4c7cb4"),
    ("E2c (edge + $k$=32, $r$=3)", 0.9223, "#2e5a9a"),
    ("E2a (edge + $k$=16, $r$=3)", 0.9227, "#1f3b73"),
]
BASELINE_RK = 0.9166


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "LaTex" / "figures" / "chapter_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [label for label, _, _ in LEVERS]
    r_at_200 = np.array([value for _, value, _ in LEVERS])
    colors = [color for _, _, color in LEVERS]

    figure, axis = plt.subplots(figsize=(5.9, 3.2))
    positions = np.arange(len(LEVERS))
    bars = axis.bar(positions, r_at_200, color=colors, edgecolor="black", linewidth=0.5)
    axis.axhline(BASELINE_RK, color="#c44e52", linestyle="--", linewidth=0.9, label="Baseline 0.9166")

    for bar, value in zip(bars, r_at_200):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axis.set_ylabel(r"$\mathrm{R}@200$ (best val)")
    axis.set_ylim(0.84, 0.94)
    axis.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axis.legend(loc="lower right", fontsize=8, frameon=False)

    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"prefilter_levers.{suffix}", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
