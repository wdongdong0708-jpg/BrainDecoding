"""论文图形的公共视觉样式。

设计原则：克制、精确、留白充足、信息层级清晰；优先矢量输出和一致的
排版、间距与视觉语义，避免无必要边框、3D、装饰性渐变和彩虹配色。
在实际可行时，图形应当在灰度环境中仍然可读。
"""

from pathlib import Path
from typing import Iterable


def configure_matplotlib() -> None:
    """配置稳定、低噪声的通用论文绘图样式。"""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figure_size(width: str = "single", aspect_ratio: float = 0.62) -> tuple[float, float]:
    """按单栏或双栏宽度返回 Figure 尺寸（英寸）。"""
    widths = {"single": 3.5, "double": 7.2}
    if width not in widths:
        raise ValueError("width 只能是 'single' 或 'double'。")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio 必须大于 0。")
    figure_width = widths[width]
    return figure_width, figure_width * aspect_ratio


def save_figure(
    figure,
    output_stem: str | Path,
    formats: Iterable[str] = ("pdf", "svg"),
    *,
    dpi: int = 300,
) -> tuple[Path, ...]:
    """用一致参数保存论文图，默认生成 PDF 和 SVG。"""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file_format in formats:
        normalized_format = file_format.lower().lstrip(".")
        output_path = stem.with_suffix(f".{normalized_format}")
        figure.savefig(output_path, format=normalized_format, dpi=dpi)
        saved_paths.append(output_path)
    return tuple(saved_paths)
