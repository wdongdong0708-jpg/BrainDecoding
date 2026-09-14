"""selector-driven 四数据集 validation audit 的边界测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding.audit import controls, runner
from braindecoding.data import pallier2025
from braindecoding.tasks.word_decoding.pallier2025 import train as pallier_training


@pytest.fixture(autouse=True)
def configured_roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )


@pytest.mark.parametrize(
    ("selector", "condition"),
    (
        ("chineseeeg2_littleprince/sub01-08/main_word", "word"),
        (
            "chineseeeg2_littleprince/sub01-08/main_context_warmstart",
            "neural_context",
        ),
        ("smn4lang/sub01-06/main_word", "word"),
        ("smn4lang/sub01-06/main_context_warmstart", "neural_context"),
        ("libribrain100/sub0/main_word", "word"),
        ("libribrain100/sub0/main_context_warmstart", "neural_context"),
        ("pallier2025/sub01-10/main_word", "word"),
        ("pallier2025/sub01-10/main_context_warmstart", "neural_context"),
    ),
)
def test_model_condition_comes_from_exact_selector_config(selector, condition):
    record, config, _ = runner._load_exact_config(selector)
    assert record["selector"] == selector
    assert runner.model_condition_from_config(config) == condition


def test_unsupported_transformer_mode_is_rejected():
    with pytest.raises(ValueError, match="grouped neural_context"):
        runner.model_condition_from_config(
            {"model": {"use_transformer": True, "context_mode": "singleton"}}
        )


def test_active_pallier_audit_binds_exact_warmstart_checkpoint():
    selector = "pallier2025/sub01-10/main_context_warmstart"
    _, config, _ = runner._load_exact_config(selector)
    checkpoint = Path(config["training"]["output_dir"]) / "best.pt"
    assert "main_context_warmstart/seed-000/best.pt" in checkpoint.as_posix()


def test_libribrain_runs_only_existing_frozen_n50_vocabulary():
    assets = runner._load_protocol_assets("libribrain100")
    assert set(assets["vocabularies"]) == {50}
    assert len(assets["vocabularies"][50]["vocabulary"]) == 50
    assert assets["story_reference"] is None
    assert assets["language"] == "en"
    assert set(assets["vocabulary_statuses"]) == {20, 100, 150}
    assert all(
        value["reason"] == "vocabulary_manifest_not_frozen"
        for value in assets["vocabulary_statuses"].values()
    )


def test_dataset_ovmi_languages_are_explicit():
    assert runner.DATASET_LANGUAGES == {
        "chineseeeg2_littleprince": "zh",
        "smn4lang": "zh",
        "libribrain100": "en",
        "pallier2025": "fr",
    }


class _RuntimeDataset:
    def __init__(self, table):
        self.table = table
        values = table["runtime_context_uid"].astype(str).tolist()
        mapping = {value: index for index, value in enumerate(dict.fromkeys(values))}
        self.sentence_indices = np.asarray(
            [mapping[value] for value in values], dtype=np.int64
        )


def _runtime_table():
    return pd.DataFrame(
        {
            "runtime_context_uid": ["rec-a|ctx-1", "rec-a|ctx-1", "rec-b|ctx-1"],
            "subject_id": ["sub-01", "sub-01", "sub-02"],
            "recording_id": ["rec-a", "rec-a", "rec-b"],
            "sentence_uid": ["ctx-1", "ctx-1", "ctx-1"],
            "记录内序号": [1, 2, 1],
        }
    )


def test_runtime_context_validation_is_recording_local():
    result = runner.validate_runtime_contexts(
        _RuntimeDataset(_runtime_table()), "runtime_context_uid", 128
    )
    assert result["cross_subject_groups"] == 0
    assert result["cross_recording_groups"] == 0
    assert result["cross_canonical_context_groups"] == 0
    assert result["groups_over_max_context_words"] == 0


def test_runtime_context_validation_rejects_cross_subject_group():
    table = _runtime_table()
    table.loc[2, "runtime_context_uid"] = "rec-a|ctx-1"
    with pytest.raises(ValueError, match="runtime context 合同失败"):
        runner.validate_runtime_contexts(
            _RuntimeDataset(table), "runtime_context_uid", 128
        )


def test_real_pallier_validation_runtime_context_contract():
    selector = "pallier2025/sub01-10/main_context_warmstart"
    _, config, adapter = runner._load_exact_config(selector)
    table = runner._validation_table(config, adapter)
    runtime = pallier2025.add_runtime_context_uid(
        table, max_context_words=config["dataset"]["max_context_words"]
    )
    result = runner.validate_runtime_contexts(
        _RuntimeDataset(runtime),
        pallier2025.RUNTIME_CONTEXT_COLUMN,
        config["dataset"]["max_context_words"],
    )
    assert result["group_count"] == 1900
    assert result["mean_words"] == pytest.approx(8.926315789473684)
    assert result["median_words"] == 8.0
    assert result["maximum_words"] == 30
    assert result["cross_subject_groups"] == 0
    assert result["cross_recording_groups"] == 0
    assert result["groups_over_max_context_words"] == 0


def test_pallier_audit_and_canonical_evaluation_loader_contract_match():
    selector = "pallier2025/sub01-10/main_context_warmstart"
    bundle = runner._build_loader_bundle(
        selector, torch.device("cpu"), load_model=False
    )
    config = bundle["config"]
    table = pallier2025.load_event_table(
        config["cache"]["event_table"],
        split="val",
        subjects=config["dataset"]["subjects"],
        trainable_only=True,
    )
    canonical_dataset = pallier_training.build_dataset(config, table)
    canonical_loader, _ = pallier_training.make_loader(
        canonical_dataset,
        config["training"],
        shuffle=False,
        group_column=pallier_training.loader_group_column(config),
    )
    assert bundle["dataset"].table["event_id"].tolist() == (
        canonical_dataset.table["event_id"].tolist()
    )
    assert bundle["dataset"].table[pallier2025.RUNTIME_CONTEXT_COLUMN].tolist() == (
        canonical_dataset.table[pallier2025.RUNTIME_CONTEXT_COLUMN].tolist()
    )
    assert np.array_equal(
        bundle["dataset"].sentence_indices,
        canonical_dataset.sentence_indices,
    )
    assert bundle["loader"].batch_sampler.groups == canonical_loader.batch_sampler.groups


def test_full_validation_clean_is_not_compared_to_smaller_core_set(tmp_path):
    config = {"training": {"output_dir": str(tmp_path)}}
    encoded = {"event_ids": ["event-1", "event-2"]}
    result = runner._saved_clean_comparison(
        config, encoded, ["event-1"], {"N50": {}}
    )
    assert result == {
        "available": False,
        "reason": "full_validation_vs_core_query_set",
    }


def test_saved_clean_comparison_accepts_machine_precision_roundoff(tmp_path):
    output = tmp_path / "run"
    evaluation = output / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "val.json").write_text(
        json.dumps(
            {
                "data": {"query_count": 2},
                "vocabularies": {
                    "50": {
                        "retrieval": {
                            "top1": 0.5,
                            "top10": 1.0,
                            "macro_top1": 0.5,
                            "macro_top10": 1.0,
                            "median_rank": 1.0,
                            "mrr": 0.75,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = {
        "N50": {
            "micro_recall_at_1": 0.5,
            "micro_recall_at_10": 1.0,
            "macro_recall_at_1": 0.5,
            "macro_recall_at_10": 1.0,
            "median_rank": 1.0,
            "mean_reciprocal_rank": 0.75 + 2e-16,
        }
    }

    result = runner._saved_clean_comparison(
        {"training": {"output_dir": str(output)}},
        {"event_ids": ["event-1", "event-2"]},
        ["event-1", "event-2"],
        metrics,
    )

    assert result["available"] is True
    assert result["exactly_equal"] is False
    assert result["numerically_equal"] is True
    assert result["absolute_tolerance"] == 1e-12
    assert result["maximum_absolute_delta"] <= 1e-12


def test_mapping_assets_are_shared_across_exact_experiments():
    assert runner._mapping_directory("pallier2025") == runner._mapping_directory(
        "pallier2025"
    )
    assert "experiment" not in runner._mapping_directory(
        "pallier2025"
    ).as_posix()


def test_mapping_only_uses_exact_selector_without_loading_checkpoint(
    tmp_path, monkeypatch
):
    selector = "pallier2025/sub01-10/main_context_warmstart"
    config = {
        "experiment": {
            "task": "word_decoding",
            "dataset": "pallier2025",
            "subject_scope": "sub01-10",
            "id": "main_context_warmstart",
            "category": "main",
        },
        "dataset": {"window_seconds": 1.0},
        "model": {"use_transformer": True, "context_mode": "grouped"},
        "training": {
            "seed": 0,
            "output_dir": str(tmp_path / "exact-warmstart-run"),
        },
        "cache": {"event_table": str(tmp_path / "events.csv")},
    }
    record = {"identity": runner.experiment_identity(config)}
    table = pd.DataFrame()
    core = {
        "query_count": 3,
        "donor_eligible_count": 3,
        "temporal_eligible_count": 3,
    }
    monkeypatch.setattr(
        runner,
        "_load_exact_config",
        lambda selected: (record, config, {"unused": selected}),
    )
    monkeypatch.setattr(runner, "_validation_table", lambda *_: table)
    monkeypatch.setattr(
        runner,
        "_build_mappings",
        lambda bundle: {"core_audit_queries.json": core},
    )
    monkeypatch.setattr(runner, "file_sha256", lambda path: "event-sha")
    monkeypatch.setattr(
        runner,
        "_load_protocol_assets",
        lambda dataset: {
            "language": "fr",
            "vocabularies": {20: {}, 50: {}, 100: {}, 150: {}},
            "vocabulary_statuses": {},
        },
    )
    monkeypatch.setattr(
        runner,
        "_build_loader_bundle",
        lambda *_args, **_kwargs: pytest.fail("mapping-only 不得加载 checkpoint"),
    )
    result = runner.run_experiment(selector, mapping_only=True)
    assert result["selector"] == selector
    assert result["model_condition"] == "neural_context"
    assert result["checkpoint"].endswith("exact-warmstart-run\\best.pt")
    assert not (tmp_path / "exact-warmstart-run").exists()


def test_validation_table_requests_only_val_split(tmp_path):
    event_path = tmp_path / "events.csv"
    event_path.write_text("fixture", encoding="utf-8")
    observed = {}

    def load_table(path, *, split, subjects, trainable_only):
        observed.update(
            path=Path(path),
            split=split,
            subjects=subjects,
            trainable_only=trainable_only,
        )
        return pd.DataFrame(
            {
                "event_id": ["event-1"],
                "subject_id": ["sub-01"],
                "recording_id": ["recording-1"],
                "normalized_word": ["mot"],
                "sentence_uid": ["context-1"],
                "split": ["val"],
                "is_trainable": [True],
                "window_start_seconds": [1.0],
                "window_stop_seconds": [2.0],
            }
        )

    config = {
        "cache": {"event_table": str(event_path)},
        "dataset": {"subjects": ["sub-01"], "window_seconds": 1.0},
    }
    result = runner._validation_table(
        config, {"load_event_table": load_table}
    )
    assert result["split"].tolist() == ["val"]
    assert observed == {
        "path": event_path,
        "split": "val",
        "subjects": ["sub-01"],
        "trainable_only": True,
    }


def test_complete_dataset_level_mappings_are_validated_and_reused(tmp_path):
    table = pd.DataFrame(
        {
            "event_id": [f"event-{index}" for index in range(4)],
            "subject_id": ["sub-01"] * 4,
            "recording_id": ["recording-01"] * 4,
            "normalized_word": ["a", "b", "c", "d"],
            "sentence_uid": ["context-01"] * 4,
            "split": ["val"] * 4,
            "is_trainable": [True] * 4,
            "window_start_seconds": [0.0, 2.0, 4.0, 6.0],
            "window_stop_seconds": [1.0, 3.0, 5.0, 7.0],
        }
    )
    assets = controls.build_control_assets(
        table,
        dataset="fixture",
        event_table_sha256="event-sha",
        status="validation_only",
    )
    controls.write_control_assets(assets, tmp_path)
    reused = runner._load_existing_mappings(
        tmp_path,
        dataset_label="fixture",
        event_table_sha256="event-sha",
    )
    assert reused == assets


def test_partial_dataset_level_mapping_set_is_rejected(tmp_path):
    (tmp_path / "clean_identity_mapping.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping 不完整"):
        runner._load_existing_mappings(
            tmp_path,
            dataset_label="fixture",
            event_table_sha256="event-sha",
        )
