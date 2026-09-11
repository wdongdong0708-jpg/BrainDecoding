import ast
from pathlib import Path

import pytest
import yaml

from braindecoding.config import (
    PROJECT_ROOT as CONFIG_PROJECT_ROOT,
    deep_merge,
    load_yaml_with_extends,
    project_path,
)
from tasks.sequence_decoding.ChineseEEG_SR.train import (
    load_config as load_chineseeeg_sr_config,
)
from tasks.word_decoding.ChineseEEG2_LittlePrince.train import 载入配置
from tasks.word_decoding.LibriBrain100.train import load_config as load_libribrain_config
from tasks.word_decoding.SMN4Lang.train import load_config as load_smn4lang_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ENTRY_PATHS = (
    PROJECT_ROOT / "tasks/sequence_decoding/ChineseEEG_SR/train.py",
    PROJECT_ROOT / "tasks/word_decoding/LibriBrain100/train.py",
    PROJECT_ROOT / "tasks/word_decoding/SMN4Lang/train.py",
    PROJECT_ROOT / "tasks/word_decoding/ChineseEEG2_LittlePrince/train.py",
)
DATASET_ROOT_CONFIGS = (
    (
        "configs/ChineseEEG2_LittlePrince.yaml",
        "${BRAINDATA_ROOT}/ChineseEEG-2/PassiveListening",
        "D:/dataset/ChineseEEG-2/PassiveListening",
    ),
    (
        "configs/ChineseEEG_SR.yaml",
        "${BRAINDATA_ROOT}/ChineseEEG",
        "D:/dataset/ChineseEEG",
    ),
    (
        "configs/LibriBrain100.yaml",
        "${BRAINDATA_ROOT}/LibriBrain100",
        "D:/dataset/LibriBrain100",
    ),
    (
        "configs/SMN4Lang.yaml",
        "${BRAINDATA_ROOT}/ds004078",
        "D:/dataset/ds004078",
    ),
    (
        "configs/SMN4Lang_gpt2.yaml",
        "${BRAINDATA_ROOT}/ds004078",
        "D:/dataset/ds004078",
    ),
)
MODEL_ROOT_CONFIGS = (
    (
        "configs/ChineseEEG2_LittlePrince.yaml",
        "${BRAINDECODING_MODEL_ROOT}/mengzi-t5-base",
        "D:/code/dascoli-word-decoding/models/mengzi-t5-base",
    ),
    (
        "configs/SMN4Lang.yaml",
        "${BRAINDECODING_MODEL_ROOT}/mengzi-t5-base",
        "D:/code/dascoli-word-decoding/models/mengzi-t5-base",
    ),
)
ALIGNMENT_CONFIGS = (
    (
        "configs/ChineseEEG2_LittlePrince.yaml",
        "outputs/女声一小王子时间戳/女声一_小王子_实际朗读时间戳.xlsx",
        (),
    ),
    (
        "configs/ChineseEEG2_LittlePrince_sub01_actual_reading_1s.yaml",
        "outputs/女声一小王子时间戳/女声一_小王子_实际朗读时间戳.xlsx",
        (),
    ),
    (
        "configs/ChineseEEG2_LittlePrince_sub05_sub08_actual_reading_1s.yaml",
        "outputs/男声一小王子时间戳/男声一_小王子_实际朗读时间戳.xlsx",
        (),
    ),
    (
        "configs/ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s.yaml",
        "outputs/女声一小王子时间戳/女声一_小王子_实际朗读时间戳.xlsx",
        (
            "outputs/女声一小王子时间戳/女声一_小王子_实际朗读时间戳.xlsx",
            "outputs/男声一小王子时间戳/男声一_小王子_实际朗读时间戳.xlsx",
        ),
    ),
)


def _legacy_project_path(value):
    """保留迁移前的项目相对路径语义，作为等价性参照。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _legacy_deep_merge(base, override):
    """保留迁移前的递归合并语义，列表和标量均由子配置替换。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _legacy_deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _legacy_load_mapping(config_path, ancestors=()):
    """独立复刻迁移前实现，避免等价测试引用新的公共实现。"""
    config_path = Path(config_path).resolve()
    if config_path in ancestors:
        raise ValueError(f"配置继承出现循环：{config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent_config = _legacy_load_mapping(
        parent_path, (*ancestors, config_path)
    )
    return _legacy_deep_merge(parent_config, config)


def _legacy_load_config(path, cache_path_keys, resolve_layout=False):
    config_path = _legacy_project_path(path)
    config = _legacy_load_mapping(config_path)
    for key in cache_path_keys:
        config["cache"][key] = str(_legacy_project_path(config["cache"][key]))
    config["training"]["output_dir"] = str(
        _legacy_project_path(config["training"]["output_dir"])
    )
    checkpoint = config["training"].get("pretrained_brain_encoder_checkpoint")
    if checkpoint:
        config["training"]["pretrained_brain_encoder_checkpoint"] = str(
            _legacy_project_path(checkpoint)
        )
    if resolve_layout:
        layout_path = config["dataset"].get("layout_path")
        if layout_path:
            config["dataset"]["layout_path"] = str(
                _legacy_project_path(layout_path)
            )
    return config


def _write_yaml(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_plain_yaml_loading(tmp_path):
    config_path = tmp_path / "plain.yaml"
    _write_yaml(config_path, "dataset:\n  name: demo\ntraining:\n  seed: 42\n")
    assert load_yaml_with_extends(config_path) == {
        "dataset": {"name": "demo"},
        "training": {"seed": 42},
    }


def test_single_relative_extends_and_override_semantics(tmp_path):
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "experiments" / "child.yaml"
    _write_yaml(
        base_path,
        "model:\n  hidden: 64\n  nested:\n    depth: 2\n    dropout: 0.1\n"
        "subjects: [sub-01, sub-02]\nseed: 1\n",
    )
    _write_yaml(
        child_path,
        "extends: ../base.yaml\nmodel:\n  nested:\n    dropout: 0.2\n"
        "subjects: [sub-03]\nseed: 7\n",
    )

    assert load_yaml_with_extends(child_path) == {
        "model": {
            "hidden": 64,
            "nested": {"depth": 2, "dropout": 0.2},
        },
        "subjects": ["sub-03"],
        "seed": 7,
    }


def test_multiple_level_extends(tmp_path):
    _write_yaml(tmp_path / "base.yaml", "a:\n  b: 1\n  c: 2\nstage: base\n")
    _write_yaml(
        tmp_path / "middle.yaml",
        "extends: base.yaml\na:\n  b: 3\nmiddle_only: true\n",
    )
    _write_yaml(
        tmp_path / "leaf.yaml",
        "extends: middle.yaml\na:\n  d: 4\nstage: leaf\n",
    )
    assert load_yaml_with_extends(tmp_path / "leaf.yaml") == {
        "a": {"b": 3, "c": 2, "d": 4},
        "stage": "leaf",
        "middle_only": True,
    }


def test_deep_merge_does_not_mutate_inputs():
    base = {"nested": {"kept": 1, "changed": 2}, "items": [1, 2], "seed": 1}
    override = {"nested": {"changed": 3}, "items": [9], "seed": 2}
    assert deep_merge(base, override) == {
        "nested": {"kept": 1, "changed": 3},
        "items": [9],
        "seed": 2,
    }
    assert base == {
        "nested": {"kept": 1, "changed": 2},
        "items": [1, 2],
        "seed": 1,
    }


def test_cyclic_extends_is_rejected(tmp_path):
    _write_yaml(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write_yaml(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="配置继承出现循环"):
        load_yaml_with_extends(tmp_path / "a.yaml")


def test_project_path_resolves_relative_and_preserves_absolute(tmp_path):
    assert CONFIG_PROJECT_ROOT == PROJECT_ROOT
    assert project_path("outputs/example") == PROJECT_ROOT / "outputs/example"
    absolute_path = tmp_path.resolve()
    assert project_path(absolute_path) == absolute_path


def test_environment_variable_expansion(tmp_path, monkeypatch):
    data_root = tmp_path / "external_data"
    monkeypatch.setenv("BRAINDATA_ROOT", str(data_root))
    config_path = tmp_path / "environment.yaml"
    _write_yaml(
        config_path,
        "dataset:\n  root: ${BRAINDATA_ROOT}/ChineseEEG-2\n",
    )
    config = load_yaml_with_extends(config_path)
    assert Path(config["dataset"]["root"]) == data_root / "ChineseEEG-2"
    assert project_path("${BRAINDATA_ROOT}/ChineseEEG-2") == (
        data_root / "ChineseEEG-2"
    )


def test_missing_environment_variable_is_rejected(tmp_path, monkeypatch):
    variable_name = "BRAINDECODING_TEST_MISSING_ROOT"
    monkeypatch.delenv(variable_name, raising=False)
    config_path = tmp_path / "missing_environment.yaml"
    _write_yaml(config_path, f"dataset:\n  root: ${{{variable_name}}}/data\n")
    with pytest.raises(ValueError, match=f"未设置的环境变量：{variable_name}"):
        load_yaml_with_extends(config_path)


@pytest.mark.parametrize(
    ("relative_path", "configured_root", "legacy_root"), DATASET_ROOT_CONFIGS
)
def test_tracked_dataset_root_uses_environment_variable(
    relative_path, configured_root, legacy_root
):
    """公共配置不再写死作者机器的数据盘。"""
    del legacy_root
    with (PROJECT_ROOT / relative_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["dataset"]["root"] == configured_root


@pytest.mark.parametrize(
    ("relative_path", "configured_root", "legacy_root"), DATASET_ROOT_CONFIGS
)
def test_tracked_dataset_root_resolves_to_legacy_path(
    relative_path, configured_root, legacy_root, monkeypatch
):
    """作者环境设置数据根目录后，最终数据位置保持不变。"""
    del configured_root
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )
    config = load_yaml_with_extends(PROJECT_ROOT / relative_path)
    assert Path(config["dataset"]["root"]) == Path(legacy_root)


def test_real_dataset_config_requires_braindata_root(monkeypatch):
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )
    monkeypatch.delenv("BRAINDATA_ROOT", raising=False)
    with pytest.raises(ValueError, match="未设置的环境变量：BRAINDATA_ROOT"):
        load_yaml_with_extends(PROJECT_ROOT / "configs/LibriBrain100.yaml")


@pytest.mark.parametrize(
    ("relative_path", "configured_model", "legacy_model"), MODEL_ROOT_CONFIGS
)
def test_tracked_text_model_uses_environment_variable(
    relative_path, configured_model, legacy_model
):
    """公共配置不再写死作者机器的文本模型目录。"""
    del legacy_model
    with (PROJECT_ROOT / relative_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["text_embedding"]["model_name"] == configured_model


@pytest.mark.parametrize(
    ("relative_path", "configured_model", "legacy_model"), MODEL_ROOT_CONFIGS
)
def test_text_model_root_resolves_to_legacy_path(
    relative_path, configured_model, legacy_model, monkeypatch
):
    """设置作者旧模型根目录后，最终模型位置保持不变。"""
    del configured_model
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )
    config = load_yaml_with_extends(PROJECT_ROOT / relative_path)
    assert Path(config["text_embedding"]["model_name"]) == Path(legacy_model)


def test_real_text_model_config_requires_model_root(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.delenv("BRAINDECODING_MODEL_ROOT", raising=False)
    with pytest.raises(
        ValueError,
        match="未设置的环境变量：BRAINDECODING_MODEL_ROOT",
    ):
        load_yaml_with_extends(
            PROJECT_ROOT / "configs/ChineseEEG2_LittlePrince.yaml"
        )


@pytest.mark.parametrize(
    ("relative_path", "alignment_path", "source_paths"), ALIGNMENT_CONFIGS
)
def test_alignment_paths_are_project_relative_and_resolve_from_any_cwd(
    relative_path,
    alignment_path,
    source_paths,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )
    monkeypatch.chdir(tmp_path)
    config = 载入配置(PROJECT_ROOT / relative_path)

    assert Path(config["dataset"]["alignment_path"]) == (
        PROJECT_ROOT / alignment_path
    )
    assert tuple(
        Path(source["alignment_path"])
        for source in config["dataset"].get("actual_reading_sources", ())
    ) == tuple(PROJECT_ROOT / path for path in source_paths)


def test_tracked_configs_do_not_contain_author_machine_drive_paths():
    for path in (PROJECT_ROOT / "configs").glob("*.yaml"):
        assert "D:/" not in path.read_text(encoding="utf-8"), path


def test_chineseeeg_sr_entry_resolves_legacy_dataset_path(monkeypatch):
    """序列任务入口也必须通过公共加载器展开数据根目录。"""
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    config = load_chineseeeg_sr_config(
        PROJECT_ROOT / "configs/ChineseEEG_SR.yaml"
    )
    assert Path(config["dataset"]["root"]) == Path("D:/dataset/ChineseEEG")


def test_training_entries_use_shared_config_primitives():
    forbidden_functions = {
        "project_path",
        "项目路径",
        "_deep_merge",
        "_递归合并",
        "_load_config_mapping",
        "_载入配置映射",
    }
    for path in TRAIN_ENTRY_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        local_functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        assert local_functions.isdisjoint(forbidden_functions)
        shared_imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "braindecoding.config"
            for alias in node.names
        }
        assert {"load_yaml_with_extends", "project_path"} <= shared_imports


@pytest.mark.parametrize(
    (
        "relative_path",
        "loader",
        "cache_path_keys",
        "resolve_layout",
        "legacy_dataset_root",
    ),
    [
        (
            "configs/LibriBrain100_1s_cnn_warm_start.yaml",
            load_libribrain_config,
            ("event_table", "meg_dir", "text_embeddings"),
            True,
            "D:/dataset/LibriBrain100",
        ),
        (
            "configs/SMN4Lang_1s_cnn_warm_start.yaml",
            load_smn4lang_config,
            ("event_table", "meg_dir", "text_embeddings"),
            True,
            "D:/dataset/ds004078",
        ),
        (
            "configs/ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_cnn_warm_start.yaml",
            载入配置,
            ("event_table", "eeg_dir", "text_embeddings"),
            False,
            "D:/dataset/ChineseEEG-2/PassiveListening",
        ),
    ],
)
def test_real_config_matches_legacy_expansion(
    relative_path,
    loader,
    cache_path_keys,
    resolve_layout,
    legacy_dataset_root,
    monkeypatch,
):
    """三类真实配置的完整展开对象必须与迁移前逐项相同。"""
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )
    expected = _legacy_load_config(
        relative_path, cache_path_keys, resolve_layout=resolve_layout
    )
    expected["dataset"]["root"] = legacy_dataset_root
    configured_model = expected.get("text_embedding", {}).get("model_name")
    if configured_model == "${BRAINDECODING_MODEL_ROOT}/mengzi-t5-base":
        expected["text_embedding"]["model_name"] = (
            "D:/code/dascoli-word-decoding/models/mengzi-t5-base"
        )
    alignment_path = expected["dataset"].get("alignment_path")
    if alignment_path:
        expected["dataset"]["alignment_path"] = str(
            _legacy_project_path(alignment_path)
        )
    for source in expected["dataset"].get("actual_reading_sources", ()):
        source["alignment_path"] = str(
            _legacy_project_path(source["alignment_path"])
        )
    actual = loader(relative_path)
    assert Path(actual["dataset"]["root"]) == Path(legacy_dataset_root)
    assert actual == expected
