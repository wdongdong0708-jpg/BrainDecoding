"""只读比较一个 canonical run 与冻结的历史结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends
from braindecoding.experiment import file_sha256, scientific_config_sha256
from braindecoding.results import canonical_retrieval


DEFAULT_MANIFEST = PROJECT_ROOT / "experiments" / "legacy_results_manifest.json"


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _record_path(entry, group, name, project_root):
    for record in entry.get(group, ()):
        path = Path(record["relative_path"])
        if path.name == name:
            return path if path.is_absolute() else Path(project_root) / path
    return None


def _comparison(canonical, legacy):
    equal = canonical == legacy if canonical is not None and legacy is not None else None
    return {"canonical": canonical, "legacy": legacy, "equal": equal}


def _numeric_comparison(canonical, legacy):
    result = _comparison(canonical, legacy)
    result["difference"] = (
        float(canonical) - float(legacy)
        if isinstance(canonical, (int, float))
        and isinstance(legacy, (int, float))
        else None
    )
    return result


def _state_shapes(path):
    if path is None or not Path(path).is_file():
        return None
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint 不含 model_state/state_dict：{path}")
    return {key: list(value.shape) for key, value in state.items()}


def _legacy_event_table_sha(summary, project_root):
    if not summary:
        return None
    if summary.get("event_table_sha256"):
        return summary["event_table_sha256"]
    value = summary.get("event_table")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(project_root) / path
    return file_sha256(path)


def _canonical_metrics(evaluation):
    if not evaluation:
        return {}
    block = evaluation.get("vocabularies", {}).get("50", {})
    return block.get("retrieval", {})


def _legacy_metrics(evaluation):
    if not evaluation:
        return {}
    return canonical_retrieval(evaluation.get("metrics", {}))


def _legacy_scientific_sha(entry, project_root):
    value = entry.get("legacy_config")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(project_root) / path
    if not path.is_file():
        return None
    return scientific_config_sha256(load_yaml_with_extends(path))


def _legacy_best_update(summary):
    value = summary.get("best_update")
    if value is not None:
        return value
    best_epoch = summary.get("best_epoch")
    for record in summary.get("history", ()):
        if record.get("epoch") == best_epoch:
            return record.get("optimizer_updates")
    return None


def compare_legacy_run(canonical_run, legacy_entry, *, project_root=PROJECT_ROOT):
    """读取两套结果并报告原值与 new-minus-legacy，不写任何源文件。"""
    canonical_run = Path(canonical_run)
    project_root = Path(project_root)
    run_manifest = _read_json(canonical_run / "run_manifest.json") or {}
    resolved_path = canonical_run / "resolved_config.yaml"
    resolved = (
        yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
        if resolved_path.is_file()
        else {}
    )
    canonical_training = _read_json(canonical_run / "training_summary.json") or {}
    canonical_evaluation = _read_json(canonical_run / "evaluation" / "val.json") or {}

    legacy_checkpoint = _record_path(
        legacy_entry, "checkpoint_files", "best.pt", project_root
    )
    legacy_summary_path = _record_path(
        legacy_entry, "summary_files", "training_summary.json", project_root
    )
    if legacy_summary_path is None:
        legacy_summary_path = _record_path(
            legacy_entry, "summary_files", "diagnostic_summary.json", project_root
        )
    legacy_evaluation_path = _record_path(
        legacy_entry, "evaluation_files", "evaluation_val.json", project_root
    )
    legacy_training = _read_json(legacy_summary_path) or {}
    legacy_evaluation = _read_json(legacy_evaluation_path) or {}
    canonical_checkpoint = canonical_run / "best.pt"

    canonical_shapes = _state_shapes(canonical_checkpoint)
    legacy_shapes = _state_shapes(legacy_checkpoint)
    canonical_metrics = _canonical_metrics(canonical_evaluation)
    legacy_metrics = _legacy_metrics(legacy_evaluation)
    metric_names = (
        "top1",
        "top10",
        "macro_top1",
        "macro_top10",
        "median_rank",
        "mrr",
    )
    canonical_selection = canonical_training.get("selection", {})
    legacy_best_epoch = legacy_training.get("best_epoch")
    legacy_best_update = _legacy_best_update(legacy_training)
    identity = run_manifest.get("experiment")
    expected_identity = legacy_entry.get("canonical_identity")
    return {
        "schema_version": 1,
        "status": "comparison_only_no_tolerance_applied",
        "legacy_run_id": legacy_entry.get("legacy_run_id"),
        "identity": _comparison(identity, expected_identity),
        "scientific_config": _comparison(
            run_manifest.get("scientific_config_sha256")
            or (scientific_config_sha256(resolved) if resolved else None),
            _legacy_scientific_sha(legacy_entry, project_root),
        ),
        "event_table_sha256": _comparison(
            (run_manifest.get("event_table") or {}).get("sha256"),
            _legacy_event_table_sha(legacy_training, project_root),
        ),
        "checkpoint_contract": {
            "canonical_path": str(canonical_checkpoint),
            "legacy_path": str(legacy_checkpoint) if legacy_checkpoint else None,
            "canonical_sha256": file_sha256(canonical_checkpoint),
            "legacy_sha256": file_sha256(legacy_checkpoint) if legacy_checkpoint else None,
            "keys_equal": (
                set(canonical_shapes) == set(legacy_shapes)
                if canonical_shapes is not None and legacy_shapes is not None
                else None
            ),
            "shapes_equal": (
                canonical_shapes == legacy_shapes
                if canonical_shapes is not None and legacy_shapes is not None
                else None
            ),
            "canonical_shapes": canonical_shapes,
            "legacy_shapes": legacy_shapes,
        },
        "validation_query_count": _comparison(
            canonical_evaluation.get("data", {}).get("query_count"),
            legacy_evaluation.get(
                "query_rows", legacy_evaluation.get("metrics", {}).get("query_count")
            ),
        ),
        "retrieval_metrics": {
            name: _numeric_comparison(
                canonical_metrics.get(name), legacy_metrics.get(name)
            )
            for name in metric_names
        },
        "selection": {
            "best_epoch": _comparison(
                canonical_selection.get("best_epoch"), legacy_best_epoch
            ),
            "best_update": _comparison(
                canonical_selection.get("best_update"), legacy_best_update
            ),
            "best_value": _numeric_comparison(
                canonical_selection.get("best_value"),
                legacy_training.get("best_score"),
            ),
        },
    }


def _manifest_entry(path, legacy_run_id):
    manifest = _read_json(path)
    matches = [
        entry for entry in manifest["runs"] if entry["legacy_run_id"] == legacy_run_id
    ]
    if len(matches) != 1:
        raise ValueError(f"legacy_run_id 必须唯一存在：{legacy_run_id}")
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_run", type=Path)
    parser.add_argument("legacy_run_id")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare_legacy_run(
        args.canonical_run,
        _manifest_entry(args.manifest, args.legacy_run_id),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
