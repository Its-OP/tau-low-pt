"""F5. Stage 3 v3 anchor training curve.

Data source:
    part/experiments/batch6_archive/B0_v3_anchor/loss_history.json

Rendered to:
    LaTex/figures/chapter_4/v3_training.pdf
    LaTex/figures/chapter_4/v3_training.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_history(history_path: Path) -> dict:
    with history_path.open() as handle:
        return json.load(handle)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    history_path = (
        root
        / "experiments"
        / "batch6_archive"
        / "B0_v3_anchor"
        / "loss_history.json"
    )
    output_dir = root.parent / "LaTex" / "figures" / "chapter_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(history_path)

    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    train_loss = np.array(history.get("train_loss", []))
    val_loss = np.array(history.get("val_loss", []))
    val_cat_100 = np.array(history.get("val_c_at_100c", []))
    if not val_cat_100.size:
        val_cat_100 = np.array(history.get("val_c_at_100", []))

    figure, axis_loss = plt.subplots(figsize=(5.8, 3.2))
    axis_loss.plot(epochs, train_loss, color="#1f3b73", linewidth=1.2, label="train loss")
    axis_loss.plot(epochs, val_loss, color="#4c7cb4", linewidth=1.2, linestyle="--", label="val loss")
    axis_loss.set_xlabel("epoch")
    axis_loss.set_ylabel("loss")
    axis_loss.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    axis_recall = axis_loss.twinx()
    if val_cat_100.size:
        axis_recall.plot(
            epochs[: val_cat_100.size],
            val_cat_100,
            color="#c44e52",
            linewidth=1.2,
            marker="o",
            markersize=3,
            label=r"val $\mathrm{C}@100$",
        )
        axis_recall.set_ylabel(r"val $\mathrm{C}@100$", color="#c44e52")
        axis_recall.tick_params(axis="y", labelcolor="#c44e52")

    lines_loss, labels_loss = axis_loss.get_legend_handles_labels()
    lines_recall, labels_recall = axis_recall.get_legend_handles_labels()
    axis_loss.legend(
        lines_loss + lines_recall,
        labels_loss + labels_recall,
        loc="upper right",
        fontsize=8,
        frameon=False,
    )

    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"v3_training.{suffix}", dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
