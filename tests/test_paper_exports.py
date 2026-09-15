import csv
import json
from pathlib import Path

import pytest
import yaml

from braindecoding import paper_exports


DATASET_CONFIGS = {
    "ChineseEEG2": ("chineseeeg2_littleprince", "sub01-08"),
    "SMN4Lang": ("smn4lang", "sub01-06"),
    "Pallier2025": ("pallier2025", "sub01-10"),
}


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _retrieval(value):
    return {
        "macro_top1": value / 4,
        "macro_top10": value,
        "median_rank": 5,
        "mrr": value / 2,
    }


def _audit_metrics(value):
    return {
        "macro_recall_at_1": value / 4,
        "macro_recall_at_10": value,
        "median_rank": 5,
        "mean_reciprocal_rank": value / 2,
    }


def _create_run(config_root, output_root, dataset, scope, experiment, model):
    identity = {
        "task": "word_decoding",
        "dataset": dataset,
        "subject_scope": scope,
        "experiment_id": experiment,
        "seed": 0,
    }
    config = {
        "experiment": {
            "task": "word_decoding",
            "dataset": dataset,
            "subject_scope": scope,
            "id": experiment,
            "category": "main",
        },
        "training": {"seed": 0},
    }
    config_path = config_root / dataset / f"{experiment}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run = output_root / "word_decoding" / dataset / scope / experiment / "seed-000"
    checkpoint_sha = f"checkpoint-{dataset}-{model}"
    _write_json(
        run / "run_manifest.json",
        {
            "experiment": identity,
            "scientific_config_sha256": f"config-{dataset}-{model}",
            "status": "completed",
        },
    )
    _write_json(
        run / "training_summary.json",
        {
            "experiment": identity,
            "checkpoint": {"best": {"sha256": checkpoint_sha}},
            "test_status": "locked_not_evaluated",
        },
    )
    vocabularies = {}
    for size in (20, 50, 100, 150):
        value = (0.4 if model == "word" else 0.6) * 20 / size
        vocabularies[str(size)] = {
            "retrieval": _retrieval(value),
        }
    _write_json(
        run / "evaluation" / "val.json",
        {
            "experiment": identity,
            "data": {"split": "val"},
            "checkpoint": {"sha256": checkpoint_sha},
            "vocabularies": vocabularies,
        },
    )

    controls = {}
    for control, value in (("clean", 0.6), ("temporal_shift", 0.25)):
        controls[control] = {
            "status": "completed",
            "results_by_vocabulary": {
                f"N{size}": _audit_metrics(value * 20 / size)
                for size in (20, 50, 100, 150)
            },
        }
    if model == "context":
        controls["structure_only"] = {
            "status": "completed",
            "results_by_vocabulary": {
                f"N{size}": _audit_metrics(10 / size)
                for size in (20, 50, 100, 150)
            },
        }
    else:
        controls["structure_only"] = {"status": "not_applicable"}
    controls["donor_swap"] = {
        "status": "completed",
        "aggregate": {
            f"N{size}": {
                "status": "completed",
                "retrieval": {
                    name: {
                        "count": 20,
                        "mean": 0.24 * 20 / size,
                        "std": 0.01,
                        "ci95_low": 0.23 * 20 / size,
                        "ci95_high": 0.25 * 20 / size,
                    }
                    for name in ("macro_top1", "macro_top10", "median_rank", "mrr")
                },
            }
            for size in (20, 50, 100, 150)
        },
    }
    _write_json(
        run / "audits" / "validation" / "summary.json",
        {
            "experiment": identity,
            "checkpoint": {"sha256": checkpoint_sha},
            "query_set": {"split": "val"},
            "controls": controls,
        },
    )
    (run / "best.pt").write_bytes(b"checkpoint must not be loaded")
    (run / "neural.npy").write_bytes(b"neural array must not be read")
    return run


@pytest.fixture
def paper_export_fixture(tmp_path, monkeypatch):
    config_root = tmp_path / "configs"
    output_root = tmp_path / "outputs"
    export_dir = tmp_path / "reports" / "exports"
    runs = []
    read_paths = []
    original_read_json = paper_exports._read_json

    for display_name, (dataset, scope) in DATASET_CONFIGS.items():
        for model, experiment in (
            ("word", "main_word"),
            ("context", "main_context_warmstart"),
        ):
            runs.append(
                _create_run(
                    config_root, output_root, dataset, scope, experiment, model
                )
            )

    libri_config = {
        "experiment": {
            "task": "word_decoding",
            "dataset": "libribrain100",
            "subject_scope": "sub0",
            "id": "main_word",
            "category": "main",
        },
        "training": {"seed": 0},
    }
    libri_path = config_root / "libribrain100" / "main_word.yaml"
    libri_path.parent.mkdir(parents=True)
    libri_path.write_text(yaml.safe_dump(libri_config), encoding="utf-8")
    libri_run = (
        output_root
        / "word_decoding"
        / "libribrain100"
        / "sub0"
        / "main_word"
        / "seed-000"
    )
    libri_run.mkdir(parents=True)
    (libri_run / "run_manifest.json").write_text("not valid JSON", encoding="utf-8")

    canonical_before = {
        path: path.read_bytes()
        for run in runs
        for path in run.rglob("*")
        if path.is_file()
    }

    def recording_read_json(path):
        read_paths.append(Path(path))
        return original_read_json(path)

    monkeypatch.setattr(paper_exports, "_read_json", recording_read_json)
    manifest = paper_exports.export_paper_results(
        export_dir,
        config_root=config_root,
        output_root=output_root,
        project_root=tmp_path,
    )
    return {
        "export_dir": export_dir,
        "manifest": manifest,
        "read_paths": read_paths,
        "canonical_before": canonical_before,
    }


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_export_contract_and_read_boundaries(paper_export_fixture):
    fixture = paper_export_fixture
    export_dir = fixture["export_dir"]
    manifest = fixture["manifest"]
    figure_rows = {
        name: _read_csv(export_dir / name)
        for name in (
            "fig2_decoding_performance.csv",
            "fig3_audit.csv",
        )
    }
    all_rows = [row for rows in figure_rows.values() for row in rows]

    assert manifest["datasets"] == list(paper_exports.PAPER_DATASETS)
    assert manifest["excluded_datasets"] == ["LibriBrain100"]
    assert manifest["available_splits"] == ["val"]
    assert {row["dataset"] for row in all_rows} == set(paper_exports.PAPER_DATASETS)
    assert all(row["dataset"] != "LibriBrain100" for row in all_rows)
    assert {row["model"] for row in all_rows} == {"word", "context"}
    assert len(figure_rows["fig2_decoding_performance.csv"]) == 24
    assert len(figure_rows["fig3_audit.csv"]) == 96

    donor = next(
        row
        for row in figure_rows["fig3_audit.csv"]
        if row["dataset"] == "ChineseEEG2"
        and row["model"] == "context"
        and row["control"] == "donor"
        and row["vocabulary_size"] == "50"
    )
    assert float(donor["macro_top10"]) == pytest.approx(0.096)
    assert float(donor["std"]) == pytest.approx(0.01)
    assert float(donor["ci_low"]) == pytest.approx(0.092)
    assert float(donor["ci_high"]) == pytest.approx(0.1)

    assert all(path.name != "test.json" for path in fixture["read_paths"])
    assert all(path.suffix == ".json" for path in fixture["read_paths"])
    assert all("libribrain100" not in str(path) for path in fixture["read_paths"])
    assert all(
        path.read_bytes() == content
        for path, content in fixture["canonical_before"].items()
    )


def test_export_module_has_no_neural_or_model_execution_path():
    source = Path(paper_exports.__file__).read_text(encoding="utf-8")
    forbidden = (
        "numpy.load",
        "np.load",
        "torch.load",
        "braindecoding.models",
        ".forward(",
    )
    assert not any(token in source for token in forbidden)


def test_current_validation_n50_matches_canonical_outputs(tmp_path):
    canonical_root = Path(__file__).resolve().parents[1] / "outputs"
    if not canonical_root.is_dir():
        pytest.skip("本地 canonical outputs 不存在")
    export_dir = tmp_path / "exports"
    paper_exports.export_paper_results(export_dir)
    fig2 = _read_csv(export_dir / "fig2_decoding_performance.csv")
    actual = {
        (row["dataset"], row["model"]): float(row["macro_top10"])
        for row in fig2
        if row["split"] == "val" and row["vocabulary_size"] == "50"
    }
    assert actual == pytest.approx(
        {
            ("ChineseEEG2", "word"): 0.2613464855726311,
            ("ChineseEEG2", "context"): 0.49680831550343074,
            ("SMN4Lang", "word"): 0.36045401728871246,
            ("SMN4Lang", "context"): 0.6371591163504814,
            ("Pallier2025", "word"): 0.35102863429377434,
            ("Pallier2025", "context"): 0.6726601879370372,
        }
    )

    fig3 = _read_csv(export_dir / "fig3_audit.csv")
    context = {
        (row["dataset"], row["control"]): float(row["macro_top10"])
        for row in fig3
        if row["split"] == "val"
        and row["model"] == "context"
        and row["vocabulary_size"] == "50"
    }
    assert context == pytest.approx(
        {
            ("ChineseEEG2", "clean"): 0.49680831550343074,
            ("ChineseEEG2", "temporal"): 0.2756690544263721,
            ("ChineseEEG2", "donor"): 0.2731685847784011,
            ("ChineseEEG2", "structure_only"): 0.2127659574468085,
            ("SMN4Lang", "clean"): 0.6371591163504814,
            ("SMN4Lang", "temporal"): 0.2500123115526898,
            ("SMN4Lang", "donor"): 0.24040856506675082,
            ("SMN4Lang", "structure_only"): 0.20833333333333334,
            ("Pallier2025", "clean"): 0.6726601879370372,
            ("Pallier2025", "temporal"): 0.26475631651809195,
            ("Pallier2025", "donor"): 0.24907184895003223,
            ("Pallier2025", "structure_only"): 0.2,
        }
    )
