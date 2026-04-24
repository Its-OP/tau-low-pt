"""F3. Hard-cut triplet survival cascade.

Data source:
    part/reports/triplet_reranking/triplet_combinatorics.md
    (K=200 and K=300 filter cascades, F1 charge through D4 rho-helicity)

Rendered to:
    LaTex/figures/chapter_4/hardcut_cascade.pdf
    LaTex/figures/chapter_4/hardcut_cascade.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Source: triplet_combinatorics.md filter-cascade tables
FILTERS = [
    "pre-cuts",
    "F1 charge",
    "F2 $\\tau$-mass",
    "F3 $a_1$-window",
    "F4 $\\rho$(770)",
    "D1 Dalitz",
    "D2 band",
    "D3 bachelor",
    "D4 helicity",
]
SURVIVAL_K200 = [100.0, 100.0, 100.0, 96.4, 66.1, 66.1, 53.6, 37.5, 26.8]
SURVIVAL_K300 = [100.0, 100.0, 100.0, 96.0, 60.0, 60.0, 48.0, 33.3, 25.3]


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "LaTex" / "figures" / "chapter_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = np.arange(len(FILTERS))
    width = 0.38

    figure, axis = plt.subplots(figsize=(6.0, 3.2))
    axis.bar(positions - width / 2, SURVIVAL_K200, width, color="#1f3b73", label=r"$K = 200$")
    axis.bar(positions + width / 2, SURVIVAL_K300, width, color="#c44e52", label=r"$K = 300$")

    axis.set_xticks(positions)
    axis.set_xticklabels(FILTERS, rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("GT triplet survival [%]")
    axis.set_ylim(0, 105)
    axis.axhline(25, color="gray", linestyle=":", linewidth=0.7)
    axis.axhline(10, color="gray", linestyle=":", linewidth=0.7)
    axis.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    axis.legend(loc="lower left", fontsize=8, frameon=False)
    axis.text(len(FILTERS) - 0.5, 11.5, "final: 10 % ($K$=200) / 12.7 % ($K$=300)",
              ha="right", va="bottom", fontsize=7.5, color="gray")

    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"hardcut_cascade.{suffix}", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
