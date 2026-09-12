"""稳定 CLI 与最终工程边界测试。"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

from braindecoding import cli
from braindecoding.catalog import discover_experiment_configs, resolve_selector
from braindecoding.config import PROJECT_ROOT


SELECTORS = {
    "chineseeeg2_littleprince/sub01-08/main_word",
    "chineseeeg2_littleprince/sub01-08/main_context",
    "smn4lang/sub01-06/main_word",
    "smn4lang/sub01-06/main_context_warmstart",
    "libribrain100/sub0/main_word",
    "libribrain100/sub0/main_context",
}


@pytest.fixture(autouse=True)
def configured_roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )


def test_root_legacy_business_modules_and_directories_are_absent():
    for name in (
        "datasets",
        "tasks",
        "models.py",
        "losses.py",
        "optimizers.py",
        "metrics.py",
        "ovmi_metrics.py",
        "run.py",
    ):
        assert not (PROJECT_ROOT / name).exists()


def test_package_production_code_has_no_legacy_root_imports():
    forbidden = {
        "datasets",
        "tasks",
        "models",
        "losses",
        "optimizers",
        "metrics",
        "ovmi_metrics",
    }
    for path in (PROJECT_ROOT / "braindecoding").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, path
            elif isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in forbidden for alias in node.names), path


def test_list_discovers_exactly_six_active_experiments(capsys):
    assert {record["selector"] for record in discover_experiment_configs()} == SELECTORS
    assert cli.main(["list"]) == 0
    output = capsys.readouterr().out
    assert all(selector.split("/")[-1] in output for selector in SELECTORS)


def test_selector_resolution_and_nearby_error():
    selector = "smn4lang/sub01-06/main_context_warmstart"
    assert resolve_selector(selector)["selector"] == selector
    with pytest.raises(ValueError, match="相近候选"):
        resolve_selector("smn4lang/sub01-06/main_contex")


def test_show_and_status_do_not_create_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_directory", lambda config: tmp_path / "never-created")
    selector = "libribrain100/sub0/main_word"
    assert cli.main(["show", selector]) == 0
    assert cli.main(["status", selector]) == 0
    assert not (tmp_path / "never-created").exists()
    assert "test: locked_not_evaluated" in capsys.readouterr().out


def test_preflight_does_not_create_run_directory(tmp_path, monkeypatch):
    import braindecoding.preflight as preflight

    selector = "libribrain100/sub0/main_word"
    record = resolve_selector(selector)
    monkeypatch.setattr(
        preflight,
        "build_preflight_report",
        lambda: {
            "experiments": [
                {
                    "config": record["relative_config"],
                    "ready": True,
                    "reasons": [],
                }
            ]
        },
    )
    monkeypatch.chdir(tmp_path)
    assert cli.main(["preflight", selector]) == 0
    assert list(tmp_path.iterdir()) == []


def test_run_missing_upstream_is_clear_and_has_no_side_effects(capsys):
    selector = "chineseeeg2_littleprince/sub01-08/main_context"
    record = resolve_selector(selector)
    output = cli.run_directory(record["raw_config"])
    assert not (output.parent.parent / "main_word" / "seed-000" / "best.pt").exists()
    assert cli.main(["run", selector]) == 2
    error = capsys.readouterr().err
    assert "Missing upstream experiment" in error
    assert "chineseeeg2_littleprince/sub01-08/main_word" in error
    assert not output.exists()


def test_cli_has_no_test_evaluation_option():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate",
                "libribrain100/sub0/main_word",
                "--split",
                "test",
            ]
        )


def test_data_check_reuses_package_builder(monkeypatch, capsys):
    import braindecoding.data.build as build

    monkeypatch.setattr(
        build,
        "check_dataset",
        lambda dataset: {"dataset": dataset, "status": "complete"},
    )
    assert cli.main(["data", "check", "smn4lang"]) == 0
    assert '"status": "complete"' in capsys.readouterr().out


def test_python_module_cli_works_from_non_project_cwd(tmp_path):
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-m", "braindecoding", "list"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "chineseeeg2_littleprince" in result.stdout


def test_installed_console_script_lists_experiments_from_non_project_cwd(tmp_path):
    executable = Path(sys.executable).parent / (
        "Scripts/brain-decoding.exe" if sys.platform == "win32" else "brain-decoding"
    )
    assert executable.is_file(), "请先执行 pip install -e ."
    result = subprocess.run(
        [executable, "list"],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "smn4lang" in result.stdout
