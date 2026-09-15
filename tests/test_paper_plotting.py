import csv
import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from paper.plotting import (  # noqa: E402
    fig2_decoding_performance,
    fig3_audit,
    make_all,
)


DATASETS = ("ChineseEEG2", "SMN4Lang", "Pallier2025")
MODELS = ("word", "context")


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def plotting_exports(tmp_path):
    export_dir = tmp_path / "reports" / "exports"
    fig2_rows = []
    fig3_rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        for model_index, model in enumerate(MODELS):
            for size in (20, 50, 100, 150):
                value = (0.4 + dataset_index / 10 + model_index / 10) * 20 / size
                fig2_rows.append(
                    {
                        "dataset": dataset,
                        "split": "val",
                        "model": model,
                        "vocabulary_size": size,
                        "available": "true",
                        "macro_top1": value / 4,
                        "macro_top10": value,
                        "median_rank": 5,
                        "mrr": value / 2,
                    }
                )
            for control, value in (
                ("clean", 0.6),
                ("temporal", 0.25),
                ("donor", 0.24),
                ("structure_only", 0.2),
            ):
                available = not (model == "word" and control == "structure_only")
                fig3_rows.append(
                    {
                        "dataset": dataset,
                        "split": "val",
                        "model": model,
                        "control": control,
                        "vocabulary_size": 50,
                        "available": str(available).lower(),
                        "macro_top1": value / 4 if available else "",
                        "macro_top10": value if available else "",
                        "median_rank": 5 if available else "",
                        "mrr": value / 2 if available else "",
                        "std": 0.01 if control == "donor" else "",
                        "ci_low": value - 0.01 if control == "donor" else "",
                        "ci_high": value + 0.01 if control == "donor" else "",
                    }
                )
    _write_csv(
        export_dir / "fig2_decoding_performance.csv",
        (
            "dataset",
            "split",
            "model",
            "vocabulary_size",
            "available",
            "macro_top1",
            "macro_top10",
            "median_rank",
            "mrr",
        ),
        fig2_rows,
    )
    _write_csv(
        export_dir / "fig3_audit.csv",
        (
            "dataset",
            "split",
            "model",
            "control",
            "vocabulary_size",
            "available",
            "macro_top1",
            "macro_top10",
            "median_rank",
            "mrr",
            "std",
            "ci_low",
            "ci_high",
        ),
        fig3_rows,
    )
    return export_dir


def test_plotting_reads_exports_and_contains_no_experiment_literals():
    plotting_dir = Path(fig2_decoding_performance.__file__).parent
    sources = "\n".join(
        path.read_text(encoding="utf-8") for path in plotting_dir.glob("*.py")
    )
    assert "reports\" / \"exports" in sources
    assert "training_summary.json" not in sources
    assert "run_manifest.json" not in sources
    assert "evaluation/val.json" not in sources
    assert "outputs/word_decoding" not in sources
    for literal in ("0.261346", "0.496808", "0.637159", "0.672660"):
        assert literal not in sources


def test_split_val_loads_and_missing_test_fails(plotting_exports):
    specifications = (
        (fig2_decoding_performance, "fig2_decoding_performance.csv"),
        (fig3_audit, "fig3_audit.csv"),
    )
    for module, filename in specifications:
        path = plotting_exports / filename
        assert module.load_data(path, "val")
        with pytest.raises(ValueError, match="不会自动回退"):
            module.load_data(path, "test")
        with pytest.raises(SystemExit) as error:
            module.main(
                ["--split", "test"],
                data_path=path,
                output_stem=plotting_exports / "unused",
            )
        assert error.value.code == 2


def test_each_figure_and_make_all_generate_pdf_and_svg(plotting_exports, tmp_path):
    individual_dir = tmp_path / "individual"
    specifications = (
        (fig2_decoding_performance, "fig2_decoding_performance.csv", "fig2"),
        (fig3_audit, "fig3_audit.csv", "fig3"),
    )
    for module, filename, stem in specifications:
        paths = module.main(
            ["--split", "val"],
            data_path=plotting_exports / filename,
            output_stem=individual_dir / stem,
        )
        assert {path.suffix for path in paths} == {".pdf", ".svg"}
        assert all(path.is_file() and path.stat().st_size > 0 for path in paths)

    all_paths = make_all.main(
        ["--split", "val"],
        export_dir=plotting_exports,
        figure_dir=tmp_path / "all",
    )
    assert len(all_paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in all_paths)
