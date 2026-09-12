"""active 流水线清理边界与清理审计记录测试。"""

import json
from pathlib import Path

import pytest

from braindecoding.config import load_yaml_with_extends
from braindecoding.experiment import resolve_experiment_config, scientific_config_sha256
from experiments.generate_cleanup_manifest import (
    ACTIVE_CONFIGS,
    CACHE_ROOTS,
    SCIENTIFIC_CONFIG_SHA256,
    _manifest_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def configured_roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )


def test_cleanup_manifest_is_the_validated_pre_delete_inventory():
    path = PROJECT_ROOT / "experiments/cleanup_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "dry_run_validated_before_cleanup"
    assert payload["manifest_sha256"] == _manifest_sha256(payload)
    assert payload["counts"] == {"archive": 218, "delete": 1866, "keep": 47}
    assert {item["action"] for item in payload["entries"]} == {
        "archive",
        "delete",
        "keep",
    }
    assert payload["legacy_result_manifest"]["run_count"] == 32


def test_only_six_active_word_configs_remain_scientifically_locked():
    runnable = []
    for relative_path in ACTIVE_CONFIGS.values():
        path = PROJECT_ROOT / relative_path
        config = resolve_experiment_config(load_yaml_with_extends(path))
        runnable.append(path)
        assert scientific_config_sha256(config) == SCIENTIFIC_CONFIG_SHA256[
            next(name for name, value in ACTIVE_CONFIGS.items() if value == relative_path)
        ]
        assert config["cache"]
        assert all(
            str(value).replace("\\", "/").startswith("derived/")
            for value in config["cache"].values()
        )
    all_runnable = []
    for path in (PROJECT_ROOT / "configs/word_decoding").rglob("*.yaml"):
        if "experiment" in load_yaml_with_extends(path):
            all_runnable.append(path)
    assert sorted(runnable) == sorted(all_runnable)


def test_active_configs_do_not_extend_legacy_or_reference_task_cache():
    for relative_path in ACTIVE_CONFIGS.values():
        path = PROJECT_ROOT / relative_path
        text = path.read_text(encoding="utf-8").replace("\\", "/")
        assert "extends: ../base.yaml" in text
        assert "tasks/word_decoding" not in text
        config = resolve_experiment_config(load_yaml_with_extends(path))
        assert "tasks/word_decoding" not in json.dumps(config, ensure_ascii=False)


def test_canonical_replacements_are_complete_before_legacy_caches_disappear():
    cleanup = json.loads(
        (PROJECT_ROOT / "experiments/cleanup_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        item["status"] == "complete"
        for item in cleanup["derived_replacements"].values()
    )
    for relative_path in CACHE_ROOTS:
        assert not (PROJECT_ROOT / relative_path).exists()


def test_smn_test_provenance_describes_preprocessing_without_evaluation():
    conditions = json.loads(
        (PROJECT_ROOT / "experiments/manifests/conditions.json").read_text(
            encoding="utf-8"
        )
    )["datasets"]["SMN4Lang"]
    assert (
        conditions["test_neural_data_status"]
        == "raw_accessed_for_deterministic_preprocessing_only"
    )
    assert conditions["test_model_evaluation"] == "not_run"
    assert conditions["test_predictions_generated"] is False
    assert conditions["test_metrics_inspected"] is False


def test_legacy_evidence_and_local_output_boundaries_are_preserved():
    legacy = json.loads(
        (PROJECT_ROOT / "experiments/legacy_results_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert legacy["run_count"] == len(legacy["runs"]) == 32
    assert (
        PROJECT_ROOT / "archive/smn4lang/sub01_legacy/SMN4Lang"
    ).is_dir()
    assert not (PROJECT_ROOT / "outputs/SMN4Lang").exists()
    assert {path.name for path in (PROJECT_ROOT / "outputs").iterdir()} == {
        "ChineseEEG1_SR",
        "ChineseEEG2_LittlePrince",
        "LibriBrain100",
    }
    assert not list((PROJECT_ROOT / "outputs").glob("pytest_*"))
