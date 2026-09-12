"""六个 active experiment 的只读训练前检查测试。"""

import json
from pathlib import Path

import pytest

from braindecoding.data.derived import stable_sha256
from braindecoding.experiment import resolved_config_sha256, run_directory
from braindecoding import preflight


@pytest.fixture(autouse=True)
def configured_roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )


def _derived_result(dataset):
    recordings = {
        "chineseeeg2_littleprince": (216, 128, 2415, 768),
        "smn4lang": (360, 306, 8685, 768),
        "libribrain100": (12, 306, 5081, 1024),
    }
    count, channels, words, dimension = recordings[dataset]
    return {
        "dataset": dataset,
        "status": "complete",
        "event_count": 1,
        "event_table_sha256": "fixture",
        "signal_coverage": {
            "expected_recordings": count,
            "signal_products": count,
            "missing": 0,
            "extra": 0,
            "channel_count": channels,
            "trainable_windows_in_range": True,
        },
        "text_contract": {
            "word_count": words,
            "shape": [words, dimension],
            "dtype": "float32",
            "train_val_only": True,
        },
        "dataset_manifest_sha256": "fixture",
        "test_model_evaluation_performed": False,
    }


def _clean_report(monkeypatch, output_root):
    monkeypatch.setattr(preflight, "git_tracked_dirty", lambda: False)
    monkeypatch.setattr(preflight, "check_dataset", _derived_result)
    return preflight.build_preflight_report(output_root=output_root)


def test_preflight_covers_exactly_six_active_configs_without_side_effects(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "outputs"
    report = _clean_report(monkeypatch, output_root)
    assert len(report["experiments"]) == 6
    identities = [tuple(item["identity"].values()) for item in report["experiments"]]
    assert len(identities) == len(set(identities)) == 6
    assert report["identity_unique"] is True
    assert report["production_legacy_references"] == []
    assert not output_root.exists()
    assert report["side_effects"] == {
        "run_directories_created": False,
        "checkpoints_generated": False,
        "model_evaluation_run": False,
        "test_model_evaluation_run": False,
    }
    assert all(item["legacy_dependency"]["status"] == "none" for item in report["experiments"])
    assert all(item["test_guard_status"]["default_split"] == "val" for item in report["experiments"])
    body = dict(report)
    expected_sha = body.pop("report_sha256")
    assert stable_sha256(body) == expected_sha


def test_context_dependencies_wait_for_same_scope_main_word(tmp_path, monkeypatch):
    report = _clean_report(monkeypatch, tmp_path / "outputs")
    by_key = {
        (item["identity"]["dataset"], item["identity"]["experiment_id"]): item
        for item in report["experiments"]
    }
    for dataset in ("chineseeeg2_littleprince", "smn4lang"):
        context = by_key[(dataset, "main_context")]
        dependency = context["dependency_status"]
        assert dependency["status"] == "waiting_for_upstream"
        assert dependency["upstream_identity"]["experiment_id"] == "main_word"
        assert dependency["upstream_identity"]["subject_scope"] == context["identity"]["subject_scope"]
        assert dependency["upstream_identity"]["seed"] == context["identity"]["seed"]
        assert context["ready"] == "waiting_for_upstream"
    assert by_key[("libribrain100", "main_context")]["dependency_status"]["status"] == "not_required"


def test_existing_upstream_checkpoint_makes_context_dependency_ready(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "outputs"
    monkeypatch.setattr(preflight, "git_tracked_dirty", lambda: False)
    monkeypatch.setattr(preflight, "check_dataset", _derived_result)
    for relative_path in preflight.ACTIVE_CONFIGS:
        config = preflight._load_resolved_config(relative_path, output_root=output_root)
        if config["experiment"]["id"] != "main_word":
            continue
        if config["experiment"]["dataset"] not in {
            "chineseeeg2_littleprince",
            "smn4lang",
        }:
            continue
        checkpoint = run_directory(config, output_root=output_root) / "best.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture")
    report = preflight.build_preflight_report(output_root=output_root)
    contexts = [
        item
        for item in report["experiments"]
        if item["identity"]["experiment_id"] == "main_context"
    ]
    assert all(item["dependency_status"]["status"] in {"ready", "not_required"} for item in contexts)
    assert all(item["ready"] is True for item in contexts)


def test_completed_run_is_reported_as_collision(tmp_path, monkeypatch):
    output_root = tmp_path / "outputs"
    relative_path = preflight.ACTIVE_CONFIGS[0]
    config = preflight._load_resolved_config(relative_path, output_root=output_root)
    run_path = run_directory(config, output_root=output_root)
    run_path.mkdir(parents=True)
    (run_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "resolved_config_sha256": resolved_config_sha256(config),
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    report = _clean_report(monkeypatch, output_root)
    record = next(
        item for item in report["experiments"] if item["config"] == relative_path
    )
    assert record["output_path"]["status"] == "collision_completed_run"
    assert record["ready"] is False


def test_git_dirty_is_a_hard_not_ready_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "git_tracked_dirty", lambda: True)
    monkeypatch.setattr(preflight, "check_dataset", _derived_result)
    report = preflight.build_preflight_report(output_root=tmp_path / "outputs")
    assert report["git"]["not_ready_reason"] == "git_dirty"
    assert all(item["ready"] is False for item in report["experiments"])
    assert all("git_dirty" in item["reasons"] for item in report["experiments"])
