import copy
import json
import re
from pathlib import Path

import pytest

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends
from braindecoding.experiment import file_sha256, warm_start_checkpoint
from experiments.freeze_legacy_results import LEGACY_RUNS, _manifest_digest


ARTIFACT_MANIFEST_PATH = (
    PROJECT_ROOT / "experiments" / "manifests" / "artifacts.json"
)
LEGACY_MANIFEST_PATH = (
    PROJECT_ROOT / "experiments" / "legacy_results_manifest.json"
)
ARCHIVE_ONLY_OUTPUTS = {
    "outputs/ChineseEEG1_SR/LittlePrince_row_retrieval",
    "outputs/SMN4Lang/word_decoding",
    "outputs/SMN4Lang/word_decoding_sentence_groups_6400updates",
}


@pytest.fixture(autouse=True)
def configured_roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_alignment_artifact_manifest_matches_local_files_byte_for_byte():
    manifest = _read_json(ARTIFACT_MANIFEST_PATH)
    artifacts = manifest["artifacts"]
    assert len(artifacts) == 16
    assert len({item["artifact_id"] for item in artifacts}) == 16
    paths = [PROJECT_ROOT / item["relative_path"] for item in artifacts]
    if not all(path.is_file() for path in paths):
        pytest.skip("本机未安装 Git 忽略的 ChineseEEG2 alignment artifacts")

    for item, path in zip(artifacts, paths):
        assert path.stat().st_size == item["size_bytes"]
        assert file_sha256(path) == item["sha256"]
        assert not (PROJECT_ROOT / item["source_legacy_path"]).exists()


def test_recorded_full_event_table_migration_is_exact():
    validation = _read_json(ARTIFACT_MANIFEST_PATH)[
        "event_table_migration_validation"
    ]
    assert validation["exact_dataframe_equal"] is True
    assert validation["raw_eeg_loaded"] is False
    assert validation["row_count_before"] == validation["row_count_after"]
    assert validation["column_count_before"] == validation["column_count_after"]
    assert (
        validation["event_table_sha256_before"]
        == validation["event_table_sha256_after"]
    )
    assert (
        validation["event_id_sequence_sha256_before"]
        == validation["event_id_sequence_sha256_after"]
    )


def test_all_configured_alignment_paths_use_artifacts_and_resolve():
    from tasks.word_decoding.ChineseEEG2_LittlePrince.train import 载入配置

    old_paths = ("outputs/女声一小王子时间戳", "outputs/男声一小王子时间戳")
    for path in (PROJECT_ROOT / "configs").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert not any(old in text for old in old_paths)

    path = (
        PROJECT_ROOT
        / "configs/word_decoding/chineseeeg2_littleprince/"
        "sub01-08/main_context.yaml"
    )
    config = 载入配置(path)
    sources = config["dataset"]["actual_reading_sources"]
    assert [source["voice_version"] for source in sources] == ["f1", "m1"]
    for source in sources:
        resolved = Path(source["alignment_path"])
        assert resolved.is_absolute()
        assert resolved.is_relative_to(PROJECT_ROOT / "artifacts")


def test_alignment_move_does_not_change_canonical_warm_start_dependency():
    path = (
        PROJECT_ROOT
        / "configs/word_decoding/chineseeeg2_littleprince/"
        "sub01-08/main_context.yaml"
    )
    config = load_yaml_with_extends(path)
    assert config["training"]["warm_start_from"] == "main_word"
    assert warm_start_checkpoint(config) == (
        PROJECT_ROOT
        / "outputs/word_decoding/chineseeeg2_littleprince/"
        "sub01-08/main_word/seed-000/best.pt"
    )


def test_legacy_manifest_covers_frozen_inventory_and_archive_only_runs():
    manifest = _read_json(LEGACY_MANIFEST_PATH)
    assert manifest["run_count"] == 32
    assert manifest["counts"]["runs"] == 32
    assert {run["legacy_output"] for run in manifest["runs"]} == {
        run["legacy_output"] for run in LEGACY_RUNS
    }
    archive_only = {
        run["legacy_output"]
        for run in manifest["runs"]
        if run["status"] == "archive_only"
    }
    assert archive_only == ARCHIVE_ONLY_OUTPUTS
    for run in manifest["runs"]:
        if run["status"] == "archive_only":
            assert run["legacy_config"] is None
            assert run["canonical_identity"] is None
            assert run["reason"] == "exact_config_unavailable"

    unsigned = copy.deepcopy(manifest)
    expected_digest = unsigned.pop("manifest_sha256")
    assert _manifest_digest(unsigned) == expected_digest


def test_legacy_manifest_records_existing_files_without_reading_their_payloads():
    manifest = _read_json(LEGACY_MANIFEST_PATH)
    records = [
        record
        for run in manifest["runs"]
        for key in (
            "checkpoint_files",
            "evaluation_files",
            "summary_files",
            "audit_files",
        )
        for record in run[key]
    ]
    assert len(records) == sum(
        manifest["counts"][key]
        for key in ("checkpoints", "evaluations", "summaries", "audits")
    )
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in records)

    output_dirs = [PROJECT_ROOT / run["legacy_output"] for run in manifest["runs"]]
    if not all(path.is_dir() for path in output_dirs):
        pytest.skip("本机没有 Git 忽略的完整历史 outputs")
    for item in records:
        path = PROJECT_ROOT / item["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == item["size_bytes"]


def test_file_hashing_is_read_only(tmp_path):
    path = tmp_path / "result.json"
    payload = b'{"metric":0.5}\n'
    path.write_bytes(payload)
    before = path.stat()

    first = file_sha256(path)
    second = file_sha256(path)
    after = path.stat()

    assert first == second
    assert path.read_bytes() == payload
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
