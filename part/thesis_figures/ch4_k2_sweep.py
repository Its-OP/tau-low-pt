"""F1. Stage 3 K2-sweep C@100 plateau.

Data source:
    part/reports/triplet_reranking/topk2_sweep_analysis_20260409.md
    (table §2 Stability analysis, last-5-epochs mean ± std)

Rendered to:
    LaTex/figures/chapter_4/k2_sweep.pdf
    LaTex/figures/chapter_4/k2_sweep.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Source: topk2_sweep_analysis_20260409.md §2
K_VALUES = [50, 60, 70, 80, 90, 100, 125, 150, 175]
CAT_100_MEAN = [0.8208, 0.8220, 0.8207, 0.8248, 0.8262, 0.8223, 0.8222, 0.8028, 0.7706]
CAT_100_STD = [0.0083, 0.0065, 0.0063, 0.0092, 0.0077, 0.0057, 0.0093, 0.0297, 0.0256]
OPERATING_POINT_K2 = 60


def main() -> None:
    output_dir = Path(__file__).resolve().parents[2] / "LaTex" / "figures" / "chapter_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = np.array(K_VALUES)
    mean_values = np.array(CAT_100_MEAN)
    std_values = np.array(CAT_100_STD)

    figure, axis = plt.subplots(figsize=(5.6, 3.0))
    axis.errorbar(
        k_values,
        mean_values,
        yerr=std_values,
        fmt="o-",
        color="#1f3b73",
        ecolor="#1f3b73",
        capsize=3,
        markersize=4,
        linewidth=1.2,
        label=r"$\mathrm{C}@100$ last-5 mean $\pm$ std",
    )
    operating_index = K_VALUES.index(OPERATING_POINT_K2)
    axis.scatter(
        [OPERATING_POINT_K2],
        [mean_values[operating_index]],
        s=90,
        color="#c44e52",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
        label=rf"$K_2 = {OPERATING_POINT_K2}$ operating point",
    )
    axis.set_xlabel(r"Stage-2 shortlist size $K_2$")
    axis.set_ylabel(r"$\mathrm{C}@100$ (last-5 mean)")
    axis.set_xticks(k_values)
    axis.set_ylim(0.76, 0.84)
    axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    axis.legend(loc="lower left", fontsize=8, frameon=False)

    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"k2_sweep.{suffix}", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
