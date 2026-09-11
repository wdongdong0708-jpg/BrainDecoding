import ast
import importlib
import pickle
from pathlib import Path

import pytest


MODULES = (
    (
        "chineseeeg2",
        "ChineseEEG2",
        "ChineseEEG2LittlePrinceWordDataset",
        (
            "构建实际朗读事件表",
            "add_event_contract",
            "审计事件表",
            "载入事件表",
            "处理后记录路径",
            "物化记录缓存",
            "_预处理签名",
            "_读取脑电记录",
        ),
    ),
    (
        "smn4lang",
        "SMN4Lang",
        "SMN4LangWordDataset",
        (
            "build_event_table",
            "add_event_contract",
            "audit_event_table",
            "load_event_table",
            "processed_recording_path",
            "materialize_recording_cache",
            "ensure_configured_text_embedding_cache",
            "_preprocessing_signature",
            "_read_raw_fif",
        ),
    ),
    (
        "libribrain",
        "LibriBrain",
        "LibriBrainWordDataset",
        (
            "build_event_table",
            "add_event_contract",
            "audit_event_table",
            "load_event_table",
            "processed_recording_path",
            "materialize_recording_cache",
            "_embedding_signature",
            "_preprocessing_signature",
        ),
    ),
)


@pytest.mark.parametrize(
    ("official_name", "legacy_name", "class_name", "interface_names"), MODULES
)
def test_legacy_dataset_modules_reexport_the_official_implementations(
    official_name, legacy_name, class_name, interface_names
):
    official = importlib.import_module(f"braindecoding.data.{official_name}")
    legacy = importlib.import_module(f"datasets.{legacy_name}")

    for name in (*interface_names, class_name):
        assert getattr(legacy, name) is getattr(official, name)
    assert getattr(official, class_name).__module__ == (
        f"braindecoding.data.{official_name}"
    )


@pytest.mark.parametrize(
    ("official_name", "legacy_name", "class_name", "interface_names"), MODULES
)
def test_old_pickle_qualified_names_still_resolve_dataset_classes(
    official_name, legacy_name, class_name, interface_names
):
    del interface_names
    official = importlib.import_module(f"braindecoding.data.{official_name}")
    old_pickle = f"cdatasets.{legacy_name}\n{class_name}\n.".encode("ascii")
    assert pickle.loads(old_pickle) is getattr(official, class_name)


def test_legacy_dataset_files_are_thin_compatibility_modules():
    project_root = Path(__file__).resolve().parents[1]
    for _, legacy_name, _, _ in MODULES:
        path = project_root / "datasets" / f"{legacy_name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        assert definitions == []


def test_production_code_uses_only_official_word_dataset_modules():
    project_root = Path(__file__).resolve().parents[1]
    legacy_names = {legacy_name for _, legacy_name, _, _ in MODULES}
    production_paths = tuple((project_root / "tasks").rglob("*.py")) + tuple(
        (project_root / "braindecoding").rglob("*.py")
    )
    for path in production_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "datasets":
                    imported = {alias.name for alias in node.names}
                    assert imported.isdisjoint(legacy_names), path
                if node.module and node.module.startswith("datasets."):
                    assert node.module.split(".")[1] not in legacy_names, path
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) >= 2 and parts[0] == "datasets":
                        assert parts[1] not in legacy_names, path
