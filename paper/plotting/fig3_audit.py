"""Figure 4：N50 clean 与严格 control 的对照。"""

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
DATA_PATH = PROJECT_ROOT / "reports" / "exports" / "fig3_audit.csv"
OUTPUT_STEM = PROJECT_ROOT / "paper" / "figures" / "fig3_audit"
DATASET_ORDER = ("ChineseEEG2", "SMN4Lang", "Pallier2025")
MODEL_ORDER = ("word", "context")
CONTROL_ORDER = ("clean", "temporal", "donor", "structure_only")
CONTROL_LABELS = {
    "clean": "Clean",
    "temporal": "Temporal",
    "donor": "Donor",
    "structure_only": "Structure\nonly",
}


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def load_data(
    path: str | Path = DATA_PATH,
    split: str = "val",
    vocabulary_size: int = 50,
) -> list[dict]:
    """读取指定 split 的 N50 audit；缺失 split 不自动回退。"""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if row["split"] != split or int(row["vocabulary_size"]) != vocabulary_size:
            continue
        selected.append(
            {
                **row,
                "vocabulary_size": int(row["vocabulary_size"]),
                "available": row["available"].lower() == "true",
                "macro_top10": _optional_float(row["macro_top10"]),
                "ci_low": _optional_float(row["ci_low"]),
                "ci_high": _optional_float(row["ci_high"]),
            }
        )
    if not selected:
        raise ValueError(
            f"导出数据中不存在 split={split!r}, vocabulary_size={vocabulary_size}，"
            "不会自动回退。"
        )
    return selected


def make_figure(
    data: list[dict],
    output_stem: str | Path = OUTPUT_STEM,
    *,
    vocabulary_size: int = 50,
) -> tuple[Path, ...]:
    """绘制 grouped bars；仅 donor 的 canonical 95% CI 使用 error bar。"""
    import matplotlib.pyplot as plt

    configure_matplotlib()
    figure, axes = plt.subplots(
        1,
        len(DATASET_ORDER),
        figsize=figure_size("double", 0.4),
        sharey=True,
        constrained_layout=True,
    )
    all_values = [
        100.0 * row["macro_top10"]
        for row in data
        if row["available"] and row["macro_top10"] is not None
    ]
    upper_limit = max(60.0, 10.0 * math.ceil((max(all_values) + 6.0) / 10.0))
    width = 0.34
    for panel_index, (axis, dataset) in enumerate(zip(axes, DATASET_ORDER)):
        dataset_rows = [row for row in data if row["dataset"] == dataset]
        for model_index, model in enumerate(MODEL_ORDER):
            style = MODEL_STYLES[model]
            offset = (model_index - 0.5) * width
            label_used = False
            for control_index, control in enumerate(CONTROL_ORDER):
                matches = [
                    row
                    for row in dataset_rows
                    if row["model"] == model
                    and row["control"] == control
                    and row["available"]
                ]
                if not matches:
                    continue
                row = matches[0]
                value = 100.0 * row["macro_top10"]
                axis.bar(
                    control_index + offset,
                    value,
                    width=width,
                    color=style["color"],
                    hatch=style["hatch"],
                    edgecolor="white",
                    linewidth=0.5,
                    label=MODEL_LABELS[model] if not label_used else None,
                    zorder=2,
                )
                label_used = True
                if control == "donor" and row["ci_low"] is not None and row["ci_high"] is not None:
                    axis.errorbar(
                        control_index + offset,
                        value,
                        yerr=[
                            [100.0 * (row["macro_top10"] - row["ci_low"])],
                            [100.0 * (row["ci_high"] - row["macro_top10"])],
                        ],
                        fmt="none",
                        ecolor="#222222",
                        elinewidth=0.8,
                        capsize=2.0,
                        capthick=0.8,
                        zorder=4,
                    )
        axis.axhline(
            100.0 * 10.0 / vocabulary_size,
            color=CHANCE_COLOR,
            linestyle=":",
            linewidth=0.9,
            alpha=0.75,
            zorder=1,
        )
        axis.set_title(dataset, pad=5)
        axis.set_xticks(range(len(CONTROL_ORDER)))
        axis.set_xticklabels([CONTROL_LABELS[name] for name in CONTROL_ORDER])
        axis.set_ylim(0, upper_limit)
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
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
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
