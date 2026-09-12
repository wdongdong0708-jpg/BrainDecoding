import copy
import importlib
import json
from pathlib import Path

import pytest
import yaml

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends
from braindecoding.experiment import (
    EXPERIMENT_CATEGORIES,
    evaluation_output_path,
    experiment_identity,
    file_sha256,
    initialize_run_directory,
    resolve_experiment_config,
    resolved_config_sha256,
    run_directory,
    scientific_config_sha256,
    update_run_status,
    warm_start_checkpoint,
)


CANONICAL_ROOTS = (
    PROJECT_ROOT / "configs" / "word_decoding",
    PROJECT_ROOT / "configs" / "sequence_decoding",
)
CANONICAL_FILES = (
    "word_decoding/chineseeeg2_littleprince/sub01-08/main_word.yaml",
    "word_decoding/chineseeeg2_littleprince/sub01-08/main_context.yaml",
    "word_decoding/smn4lang/sub01-06/main_word.yaml",
    "word_decoding/smn4lang/sub01-06/main_context.yaml",
    "word_decoding/libribrain100/sub0/main_word.yaml",
    "word_decoding/libribrain100/sub0/main_context.yaml",
    "sequence_decoding/chineseeeg1_sr/sub04-10_sub13-14/historical_row_retrieval_fourier_subject.yaml",
    "sequence_decoding/chineseeeg1_sr/sub04-10_sub13-14/historical_closed_set_loso.yaml",
)
ACTIVE_SCIENTIFIC_SHA256 = {
    "word_decoding/chineseeeg2_littleprince/sub01-08/main_word.yaml": "3117f60305b0aa6de8dd930e896ab9fa3ab26829acbab28a3cdb89e9f863573e",
    "word_decoding/chineseeeg2_littleprince/sub01-08/main_context.yaml": "3951b8a5d1936265da3d506ada1e74aeb5e2a34ae4466a4cb6fbdf70150d6387",
    "word_decoding/smn4lang/sub01-06/main_word.yaml": "40fba01a86059b0b1299498b0bbff50aca2b8b0080995936604afbf84103ca1a",
    "word_decoding/smn4lang/sub01-06/main_context.yaml": "91cb836820b6d1e49077396f19eea013f703c20e2ee0b68d62f5501b3677a9d2",
    "word_decoding/libribrain100/sub0/main_word.yaml": "af6c317dcc0b8e697504d69f887765762bf20394f7f1f7786470dfc6de951617",
    "word_decoding/libribrain100/sub0/main_context.yaml": "bd27d21f8a7ea54c380483a019596223eee5144625a927322319bdd0e6d62385",
}
WARM_STARTS = {
    "word_decoding/chineseeeg2_littleprince/sub01-08/main_context.yaml": "main_word",
    "word_decoding/smn4lang/sub01-06/main_context.yaml": "main_word",
}
ARCHIVE_ONLY_OUTPUTS = (
    "outputs/ChineseEEG1_SR/LittlePrince_row_retrieval",
    "outputs/SMN4Lang/word_decoding",
    "outputs/SMN4Lang/word_decoding_sentence_groups_6400updates",
)


@pytest.fixture(autouse=True)
def configured_roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )


def _canonical_path(relative):
    return PROJECT_ROOT / "configs" / relative


def _canonical_files():
    return tuple(_canonical_path(relative) for relative in CANONICAL_FILES)


def _without_identity_and_paths(config):
    cleaned = copy.deepcopy(config)
    cleaned.pop("experiment", None)
    cleaned.pop("run_section", None)
    training = cleaned.get("training", {})
    training.pop("output_dir", None)
    training.pop("warm_start_from", None)
    training.pop("pretrained_brain_encoder_checkpoint", None)
    cleaned.get("closed_set_diagnostic", {}).pop("output_dir", None)
    return cleaned


def _science_signature(config):
    cleaned = _without_identity_and_paths(config)
    cleaned.pop("provenance", None)
    if config.get("run_section"):
        cleaned["run_section"] = config["run_section"]
    return resolved_config_sha256(cleaned)


def test_canonical_inventory_and_identity_are_complete_and_unique():
    assert len(CANONICAL_FILES) == 8
    identities = []
    for path in _canonical_files():
        config = load_yaml_with_extends(path)
        identity = experiment_identity(config)
        assert config["experiment"]["category"] in EXPERIMENT_CATEGORIES
        identities.append(tuple(identity.values()))
    assert len(identities) == len(set(identities))


def test_canonical_yaml_never_writes_output_dir_or_parameterized_ids():
    forbidden = ("6400updates", "768", "1024")
    for root in CANONICAL_ROOTS:
        for path in root.rglob("*.yaml"):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            assert "output_dir" not in path.read_text(encoding="utf-8")
            if path.name == "base.yaml":
                assert "experiment" not in raw
                continue
            experiment_id = raw["experiment"]["id"]
            assert not any(token in experiment_id for token in forbidden)


def test_output_paths_are_derived_from_identity_without_collisions():
    paths = []
    for path in _canonical_files():
        config = load_yaml_with_extends(path)
        output = run_directory(config)
        identity = experiment_identity(config)
        assert output == (
            PROJECT_ROOT
            / "outputs"
            / identity["task"]
            / identity["dataset"]
            / identity["subject_scope"]
            / identity["experiment_id"]
            / f"seed-{identity['seed']:03d}"
        )
        assert str(output).endswith("seed-000") or str(output).endswith("seed-042")
        paths.append(output)
    assert len(paths) == len(set(paths))


@pytest.mark.parametrize("canonical,expected", ACTIVE_SCIENTIFIC_SHA256.items())
def test_active_scientific_config_matches_pre_cleanup_contract(canonical, expected):
    current = resolve_experiment_config(
        load_yaml_with_extends(_canonical_path(canonical))
    )
    assert scientific_config_sha256(current) == expected


def test_no_two_canonical_configs_have_the_same_scientific_role():
    signatures = []
    for path in _canonical_files():
        config = load_yaml_with_extends(path)
        identity = experiment_identity(config)
        signatures.append(
            (
                identity["task"],
                identity["dataset"],
                identity["subject_scope"],
                _science_signature(config),
            )
        )
    assert len(signatures) == len(set(signatures))


@pytest.mark.parametrize("canonical,source_id", WARM_STARTS.items())
def test_warm_start_dependencies_use_same_scope_and_seed(canonical, source_id):
    config = load_yaml_with_extends(_canonical_path(canonical))
    checkpoint = warm_start_checkpoint(config)
    identity = experiment_identity(config)
    assert checkpoint == (
        PROJECT_ROOT
        / "outputs"
        / identity["task"]
        / identity["dataset"]
        / identity["subject_scope"]
        / source_id
        / f"seed-{identity['seed']:03d}"
        / "best.pt"
    )
    resolved = resolve_experiment_config(config)
    assert Path(resolved["training"]["pretrained_brain_encoder_checkpoint"]) == (
        checkpoint
    )


def test_hash_is_stable_across_dictionary_key_order():
    left = {"experiment": {"task": "x", "id": "y"}, "training": {"seed": 0}}
    right = {"training": {"seed": 0}, "experiment": {"id": "y", "task": "x"}}
    assert resolved_config_sha256(left) == resolved_config_sha256(right)


def test_evaluation_layout_changes_only_for_canonical_configs(tmp_path):
    canonical = resolve_experiment_config(_minimal_config(), output_root=tmp_path)
    assert evaluation_output_path(canonical, "val") == (
        tmp_path / "word_decoding/demo/sub01/main_word/seed-000/evaluation/val.json"
    )
    assert evaluation_output_path(canonical, "val", smoke=True).name == (
        "val_smoke.json"
    )
    legacy = {"training": {"output_dir": str(tmp_path / "legacy")}}
    assert evaluation_output_path(legacy, "val") == (
        tmp_path / "legacy/evaluation_val.json"
    )


def test_identity_rejects_path_components_and_non_integer_seed():
    config = _minimal_config()
    config["experiment"]["id"] = "../main_word"
    with pytest.raises(ValueError, match="身份字段"):
        experiment_identity(config)
    config = _minimal_config()
    config["training"]["seed"] = 0.5
    with pytest.raises(ValueError, match="非负整数"):
        experiment_identity(config)


def _minimal_config():
    return {
        "experiment": {
            "task": "word_decoding",
            "dataset": "demo",
            "subject_scope": "sub01",
            "id": "main_word",
            "category": "main",
        },
        "training": {"seed": 0},
        "model": {"hidden": 4},
    }


def test_run_assets_and_overwrite_guards(tmp_path):
    config = _minimal_config()
    event_table = tmp_path / "events.csv"
    protocol = tmp_path / "protocol.json"
    vocabulary = tmp_path / "vocabulary.json"
    event_table.write_text("event_id\n1\n", encoding="utf-8")
    protocol.write_text('{"status":"frozen"}\n', encoding="utf-8")
    vocabulary.write_text('{"vocabulary":["word"]}\n', encoding="utf-8")
    config["cache"] = {"event_table": str(event_table)}
    output_dir, manifest = initialize_run_directory(
        config,
        ["python", "train.py"],
        allow_dirty=True,
        output_root=tmp_path,
        protocol_manifests=(protocol,),
        vocabulary_manifests=(vocabulary,),
    )
    assert output_dir == tmp_path / "word_decoding/demo/sub01/main_word/seed-000"
    assert (output_dir / "resolved_config.yaml").is_file()
    assert (output_dir / "run_manifest.json").is_file()
    assert (output_dir / "evaluation").is_dir()
    assert (output_dir / "audits/validation").is_dir()
    assert manifest["status"] == "running"
    assert manifest["event_table"] == {
        "path": str(event_table),
        "sha256": file_sha256(event_table),
    }
    assert manifest["protocol_manifests"][0]["sha256"] == file_sha256(protocol)
    assert manifest["vocabulary_manifests"][0]["sha256"] == file_sha256(
        vocabulary
    )
    assert manifest["resolved_config_sha256"] == resolved_config_sha256(
        resolve_experiment_config(config, output_root=tmp_path)
    )

    with pytest.raises(FileExistsError, match="显式 resume"):
        initialize_run_directory(
            config, "python train.py", allow_dirty=True, output_root=tmp_path
        )
    initialize_run_directory(
        config,
        "python train.py",
        resume=True,
        allow_dirty=True,
        output_root=tmp_path,
    )
    update_run_status(output_dir, "completed")
    with pytest.raises(FileExistsError, match="已经完成"):
        initialize_run_directory(
            config,
            "python train.py",
            resume=True,
            allow_dirty=True,
            output_root=tmp_path,
        )


def test_run_directory_rejects_missing_manifest_and_changed_config(tmp_path):
    config = _minimal_config()
    output_dir = run_directory(config, output_root=tmp_path)
    output_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="缺少 run_manifest"):
        initialize_run_directory(
            config, "train", allow_dirty=True, output_root=tmp_path
        )

    other_root = tmp_path / "other"
    initialize_run_directory(
        config, "train", allow_dirty=True, output_root=other_root
    )
    changed = copy.deepcopy(config)
    changed["model"]["hidden"] = 8
    with pytest.raises(FileExistsError, match="配置 SHA 不同"):
        initialize_run_directory(
            changed,
            "train",
            resume=True,
            allow_dirty=True,
            output_root=other_root,
        )


def test_archive_only_orphans_have_no_runnable_canonical_identity():
    canonical_outputs = {
        run_directory(load_yaml_with_extends(path)).relative_to(PROJECT_ROOT).as_posix()
        for path in _canonical_files()
    }
    for orphan in ARCHIVE_ONLY_OUTPUTS:
        assert not any(orphan in output for output in canonical_outputs)


def test_task_loaders_inject_canonical_output_and_warm_start_paths():
    from tasks.sequence_decoding.ChineseEEG_SR.train import (
        load_config as load_sequence,
        validate_run_section,
    )
    from tasks.word_decoding.ChineseEEG2_LittlePrince.train import 载入配置
    from tasks.word_decoding.LibriBrain100.train import load_config as load_libribrain
    from tasks.word_decoding.SMN4Lang.train import load_config as load_smn4lang

    cases = (
        (
            载入配置,
            "word_decoding/chineseeeg2_littleprince/sub01-08/main_context.yaml",
            "word_decoding/chineseeeg2_littleprince/sub01-08/main_context/seed-000",
            "word_decoding/chineseeeg2_littleprince/sub01-08/main_word/seed-000/best.pt",
        ),
        (
            load_smn4lang,
            "word_decoding/smn4lang/sub01-06/main_context.yaml",
            "word_decoding/smn4lang/sub01-06/main_context/seed-000",
            "word_decoding/smn4lang/sub01-06/main_word/seed-000/best.pt",
        ),
    )
    for loader, relative, output_suffix, checkpoint_suffix in cases:
        config = loader(PROJECT_ROOT / "configs" / relative)
        assert Path(config["training"]["output_dir"]) == PROJECT_ROOT / "outputs" / output_suffix
        assert Path(config["training"]["pretrained_brain_encoder_checkpoint"]) == (
            PROJECT_ROOT / "outputs" / checkpoint_suffix
        )

    libribrain = load_libribrain(
        PROJECT_ROOT
        / "configs/word_decoding/libribrain100/sub0/main_context.yaml"
    )
    assert Path(libribrain["training"]["output_dir"]) == (
        PROJECT_ROOT
        / "outputs/word_decoding/libribrain100/sub0/main_context/seed-000"
    )
    assert "pretrained_brain_encoder_checkpoint" not in libribrain["training"]

    sequence = load_sequence(
        PROJECT_ROOT
        / "configs/sequence_decoding/chineseeeg1_sr/sub04-10_sub13-14/"
        "historical_closed_set_loso.yaml"
    )
    expected = (
        PROJECT_ROOT
        / "outputs/sequence_decoding/chineseeeg1_sr/sub04-10_sub13-14/"
        "historical_closed_set_loso/seed-042"
    )
    assert Path(sequence["training"]["output_dir"]) == expected
    assert Path(sequence["closed_set_diagnostic"]["output_dir"]) == expected
    validate_run_section(sequence, closed_set_loso=True)
    with pytest.raises(ValueError, match="不能运行 training"):
        validate_run_section(sequence, closed_set_loso=False)


def test_canonical_base_files_are_explicitly_non_runnable():
    bases = [path for root in CANONICAL_ROOTS for path in root.rglob("base.yaml")]
    assert len(bases) == 4
    for path in bases:
        config = load_yaml_with_extends(path)
        with pytest.raises(ValueError, match="缺少 experiment"):
            experiment_identity(config)


@pytest.mark.parametrize(
    "module_name,loader_name,runner_name,printer_name",
    (
        (
            "tasks.word_decoding.ChineseEEG2_LittlePrince.train",
            "载入配置",
            "执行训练",
            "打印训练摘要",
        ),
        (
            "tasks.word_decoding.SMN4Lang.train",
            "load_config",
            "run_training",
            "print_training_summary",
        ),
        (
            "tasks.word_decoding.LibriBrain100.train",
            "load_config",
            "run_training",
            "print_training_summary",
        ),
        (
            "tasks.sequence_decoding.ChineseEEG_SR.train",
            "load_config",
            "run_training",
            "print_training_summary",
        ),
    ),
)
def test_canonical_cli_records_running_then_completed_without_real_training(
    monkeypatch,
    tmp_path,
    module_name,
    loader_name,
    runner_name,
    printer_name,
):
    module = importlib.import_module(module_name)
    config = _minimal_config()
    if "sequence_decoding" in module_name:
        config["run_section"] = "training"
        config["training"]["top_ks"] = [1, 10]
    statuses = []

    monkeypatch.setattr(module, loader_name, lambda path: copy.deepcopy(config))
    monkeypatch.setattr(
        module,
        "initialize_run_directory",
        lambda loaded, command: (tmp_path / "run", {"status": "running"}),
    )
    monkeypatch.setattr(
        module,
        "update_run_status",
        lambda output, status: statuses.append((output, status)),
    )
    monkeypatch.setattr(module, runner_name, lambda loaded, **kwargs: {})
    monkeypatch.setattr(module, printer_name, lambda *args, **kwargs: None)

    module.main(["--config", "unused.yaml"])
    assert statuses == [(tmp_path / "run", "completed")]


def test_canonical_cli_marks_failed_run_without_real_training(monkeypatch, tmp_path):
    module = importlib.import_module("tasks.word_decoding.SMN4Lang.train")
    config = _minimal_config()
    statuses = []
    monkeypatch.setattr(module, "load_config", lambda path: copy.deepcopy(config))
    monkeypatch.setattr(
        module,
        "initialize_run_directory",
        lambda loaded, command: (tmp_path / "run", {"status": "running"}),
    )
    monkeypatch.setattr(
        module,
        "update_run_status",
        lambda output, status: statuses.append((output, status)),
    )

    def fail_without_training(loaded, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(module, "run_training", fail_without_training)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        module.main(["--config", "unused.yaml"])
    assert statuses == [(tmp_path / "run", "failed")]
