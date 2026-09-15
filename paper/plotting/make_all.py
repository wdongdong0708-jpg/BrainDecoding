"""一次生成当前正文使用的 Figure 2 和 Figure 3。"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import fig2_decoding_performance, fig3_audit
except ImportError:  # 直接运行本文件时使用。
    import fig2_decoding_performance
    import fig3_audit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = PROJECT_ROOT / "reports" / "exports"
FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"


def make_all(
    split: str,
    *,
    export_dir: str | Path = EXPORT_DIR,
    figure_dir: str | Path = FIGURE_DIR,
) -> tuple[Path, ...]:
    export_dir = Path(export_dir)
    figure_dir = Path(figure_dir)
    outputs = []
    specifications = (
        (
            fig2_decoding_performance,
            export_dir / "fig2_decoding_performance.csv",
            figure_dir / "fig2_decoding_performance",
        ),
        (
            fig3_audit,
            export_dir / "fig3_audit.csv",
            figure_dir / "fig3_audit",
        ),
    )
    for module, data_path, output_stem in specifications:
        data = module.load_data(data_path, split)
        outputs.extend(module.make_figure(data, output_stem))
    return tuple(outputs)


def main(
    argv: list[str] | None = None,
    *,
    export_dir: str | Path = EXPORT_DIR,
    figure_dir: str | Path = FIGURE_DIR,
) -> tuple[Path, ...]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    args = parser.parse_args(argv)
    try:
        return make_all(args.split, export_dir=export_dir, figure_dir=figure_dir)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
