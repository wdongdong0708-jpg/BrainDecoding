"""Figure 2：三个正式数据集上的词汇解码性能。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

try:
    from .paper_style import (
        CHANCE_COLOR,
        MODEL_LABELS,
        MODEL_STYLES,
        configure_matplotlib,
        figure_size,
        save_figure,
    )
except ImportError:  # 直接运行本文件时使用。
    from paper_style import (
        CHANCE_COLOR,
        MODEL_LABELS,
        MODEL_STYLES,
        configure_matplotlib,
        figure_size,
        save_figure,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "reports" / "exports" / "fig2_decoding_performance.csv"
OUTPUT_STEM = PROJECT_ROOT / "paper" / "figures" / "fig2_decoding_performance"
DATASET_ORDER = ("ChineseEEG2", "SMN4Lang", "Pallier2025")
MODEL_ORDER = ("word", "context")
VOCABULARY_SIZES = (20,)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def load_data(path: str | Path = DATA_PATH, split: str = "val") -> list[dict]:
    """读取一个明确 split；不存在时拒绝自动回退。"""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if row["split"] != split:
            continue
        selected.append(
            {
                **row,
                "vocabulary_size": int(row["vocabulary_size"]),
                "available": row["available"].lower() == "true",
                "macro_top10": _optional_float(row["macro_top10"]),
            }
        )
    if not selected:
        raise ValueError(f"导出数据中不存在 split={split!r}，不会自动回退。")
    return selected


def make_figure(
    data: list[dict], output_stem: str | Path = OUTPUT_STEM
) -> tuple[Path, ...]:
    """生成三横向 panel、共享纵轴的 Macro Top-10 曲线图。"""
    import matplotlib.pyplot as plt

    configure_matplotlib()
    figure, axes = plt.subplots(
        1,
        len(DATASET_ORDER),
        figsize=figure_size("double", 0.36),
        sharey=True,
        constrained_layout=True,
    )
    chance = [100.0 * 10.0 / size for size in VOCABULARY_SIZES]
    for panel_index, (axis, dataset) in enumerate(zip(axes, DATASET_ORDER)):
        dataset_rows = [row for row in data if row["dataset"] == dataset]
        for model in MODEL_ORDER:
            by_size = {
                row["vocabulary_size"]: row["macro_top10"]
                for row in dataset_rows
                if row["model"] == model and row["available"]
            }
            values = [
                100.0 * by_size[size] if size in by_size else math.nan
                for size in VOCABULARY_SIZES
            ]
            style = MODEL_STYLES[model]
            axis.plot(
                VOCABULARY_SIZES,
                values,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                markersize=4.2,
                label=MODEL_LABELS[model],
            )
        axis.plot(
            VOCABULARY_SIZES,
            chance,
            color=CHANCE_COLOR,
            linestyle=":",
            linewidth=0.9,
            alpha=0.75,
            label="Chance (10/N)",
        )
        axis.set_title(dataset, pad=5)
        axis.set_xlabel("Vocabulary size")
        axis.set_xticks(VOCABULARY_SIZES)
        axis.set_ylim(0, 100)
        axis.tick_params(direction="out", length=3)
        axis.text(
            -0.14,
            1.04,
            chr(ord("a") + panel_index),
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
    axes[0].set_ylabel("Macro Top-10 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    saved = save_figure(figure, output_stem)
    plt.close(figure)
    return saved


def main(
    argv: list[str] | None = None,
    *,
    data_path: str | Path = DATA_PATH,
    output_stem: str | Path = OUTPUT_STEM,
) -> tuple[Path, ...]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    args = parser.parse_args(argv)
    try:
        data = load_data(data_path, args.split)
    except ValueError as error:
        parser.error(str(error))
    return make_figure(data, output_stem)


if __name__ == "__main__":
    main()
