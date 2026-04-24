"""F4. Stage 3 Block-1 embedding Batch-2 ablation.

Data source:
    part/reports/couple_ranking_overview.md §5 (Batch 2 ablation summary)
    ΔC@100 vs v2 concat baseline, last-5 mean:
        A1 raw InferSent        +0.0003
        A2 symmetric            -0.0062
        A3 bilinear LRB r=8     -0.0005
        A4 projected p=16       +0.0004
        A5 projected p=32       +0.0010 (winner, promoted to v3)

Rendered to:
    LaTex/figures/chapter_4/couple_embed.pdf
    LaTex/figures/chapter_4/couple_embed.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Source: couple_ranking_overview.md §5.3-§5.7
VARIANTS = [
    ("A2 symmetric pool", -0.0062, "#c44e52"),
    ("A3 bilinear LRB ($r$=8)", -0.0005, "#d88489"),
    ("A1 raw InferSent", 0.0003, "#8fa9d6"),
    ("A4 projected ($p$=16)", 0.0004, "#4c7cb4"),
    ("A5 projected ($p$=32)", 0.0010, "#1f3b73"),
]


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "LaTex" / "figures" / "chapter_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [label for label, _, _ in VARIANTS]
    deltas = np.array([value for _, value, _ in VARIANTS])
    colors = [color for _, _, color in VARIANTS]

    figure, axis = plt.subplots(figsize=(5.6, 3.0))
    positions = np.arange(len(VARIANTS))
    bars = axis.bar(positions, deltas * 100, color=colors, edgecolor="black", linewidth=0.5)
    axis.axhline(0, color="black", linewidth=0.7)

    for bar, delta in zip(bars, deltas):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            delta * 100 + (0.05 if delta >= 0 else -0.05),
            f"{delta*100:+.2f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=7.5,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axis.set_ylabel(r"$\Delta \mathrm{C}@100$ vs v2 concat [pp $\times 10$]")
    axis.set_ylim(-0.9, 0.25)
    axis.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"couple_embed.{suffix}", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
