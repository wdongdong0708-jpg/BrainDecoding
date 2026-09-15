"""从 canonical 单次实验 JSON 生成扁平的论文绘图数据。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from braindecoding.catalog import resolve_selector
from braindecoding.config import PROJECT_ROOT
from braindecoding.experiment import file_sha256, run_directory


PAPER_DATASETS = ("ChineseEEG2", "SMN4Lang", "Pallier2025")

_DATASET_RUNS = {
    "ChineseEEG2": ("chineseeeg2_littleprince", "sub01-08"),
    "SMN4Lang": ("smn4lang", "sub01-06"),
    "Pallier2025": ("pallier2025", "sub01-10"),
}
_MODEL_RUNS = {"word": "main_word", "context": "main_context_warmstart"}
_CONTROL_NAMES = {
    "clean": "clean",
    "temporal_shift": "temporal",
    "donor_swap": "donor",
    "structure_only": "structure_only",
}

FIG2_FIELDS = (
    "dataset",
    "split",
    "model",
    "vocabulary_size",
    "available",
    "macro_top1",
    "macro_top10",
    "median_rank",
    "mrr",
)
FIG3_FIELDS = (
    "dataset",
    "split",
    "model",
    "vocabulary_size",
    "reference",
    "available",
    "ovmi_bits",
    "coverage",
    "in_vocab_information_bits",
    "reason",
)
FIG4_FIELDS = (
    "dataset",
    "split",
    "model",
    "control",
    "vocabulary_size",
    "available",
    "macro_top1",
    "macro_top10",
    "median_rank",
    "mrr",
    "std",
    "ci_low",
    "ci_high",
)


def paper_selectors() -> tuple[tuple[str, str, str], ...]:
    """返回按论文显示顺序排列的 ``(dataset, model, selector)``。"""
    selectors = []
    for display_name in PAPER_DATASETS:
        dataset, subject_scope = _DATASET_RUNS[display_name]
        for model, experiment_id in _MODEL_RUNS.items():
            selectors.append(
                (
                    display_name,
                    model,
                    f"{dataset}/{subject_scope}/{experiment_id}",
                )
            )
    return tuple(selectors)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 canonical JSON：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"canonical JSON 顶层必须是对象：{path}")
    return value


def _evaluation_rows(dataset: str, model: str, evaluation: dict) -> list[dict]:
    split = evaluation.get("data", {}).get("split")
    rows = []
    for size_text, block in evaluation.get("vocabularies", {}).items():
        size = int(size_text)
        retrieval = block.get("retrieval", {})
        macro_top10 = retrieval.get("macro_top10")
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "model": model,
                "vocabulary_size": size,
                "available": macro_top10 is not None,
                "macro_top1": retrieval.get("macro_top1"),
                "macro_top10": macro_top10,
                "median_rank": retrieval.get("median_rank"),
                "mrr": retrieval.get("mrr"),
            }
        )
    return sorted(rows, key=lambda row: row["vocabulary_size"])


def _information_rows(dataset: str, model: str, evaluation: dict) -> list[dict]:
    split = evaluation.get("data", {}).get("split")
    rows = []
    for size_text, block in evaluation.get("vocabularies", {}).items():
        for reference in ("story", "story_specific"):
            result = block.get("ovmi", {}).get(reference)
            if result is None:
                continue
            available = result.get("available") is True
            reason = result.get("reason")
            if not available and reason is None:
                reason = result.get("status", "not_run")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "model": model,
                    "vocabulary_size": int(size_text),
                    "reference": reference,
                    "available": available,
                    "ovmi_bits": result.get("score_bits") if available else None,
                    "coverage": result.get("coverage") if available else None,
                    "in_vocab_information_bits": (
                        result.get("in_vocab_information_bits") if available else None
                    ),
                    "reason": None if available else reason,
                }
            )
    return sorted(rows, key=lambda row: (row["vocabulary_size"], row["reference"]))


def _direct_audit_metrics(block: dict) -> dict:
    return {
        "macro_top1": block.get("macro_recall_at_1"),
        "macro_top10": block.get("macro_recall_at_10"),
        "median_rank": block.get("median_rank"),
        "mrr": block.get("mean_reciprocal_rank"),
    }


def _mean(metric: dict) -> float | None:
    return metric.get("mean") if isinstance(metric, dict) else None


def _audit_rows(dataset: str, model: str, audit: dict) -> list[dict]:
    split = audit.get("query_set", {}).get("split")
    controls = audit.get("controls", {})
    vocabulary_sizes = set()
    for source_name in ("clean", "temporal_shift", "structure_only"):
        for key in controls.get(source_name, {}).get("results_by_vocabulary", {}):
            if str(key).startswith("N"):
                vocabulary_sizes.add(int(str(key)[1:]))
    for key in controls.get("donor_swap", {}).get("aggregate", {}):
        if str(key).startswith("N"):
            vocabulary_sizes.add(int(str(key)[1:]))

    rows = []
    for size in sorted(vocabulary_sizes):
        vocabulary_key = f"N{size}"
        for source_name in ("clean", "temporal_shift", "structure_only"):
            control = controls.get(source_name, {})
            block = control.get("results_by_vocabulary", {}).get(vocabulary_key, {})
            metrics = _direct_audit_metrics(block)
            available = (
                control.get("status") == "completed"
                and metrics["macro_top10"] is not None
            )
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "model": model,
                    "control": _CONTROL_NAMES[source_name],
                    "vocabulary_size": size,
                    "available": available,
                    **{key: value if available else None for key, value in metrics.items()},
                    "std": None,
                    "ci_low": None,
                    "ci_high": None,
                }
            )

        donor = controls.get("donor_swap", {}).get("aggregate", {}).get(
            vocabulary_key, {}
        )
        retrieval = donor.get("retrieval", {})
        macro_top10 = retrieval.get("macro_top10", {})
        available = donor.get("status") == "completed" and _mean(macro_top10) is not None
        if available and macro_top10.get("count") != 20:
            raise ValueError(
                f"{dataset}/{model}/{vocabulary_key} donor aggregate 不是 20 seeds。"
            )
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "model": model,
                "control": _CONTROL_NAMES["donor_swap"],
                "vocabulary_size": size,
                "available": available,
                "macro_top1": _mean(retrieval.get("macro_top1", {})) if available else None,
                "macro_top10": _mean(macro_top10) if available else None,
                "median_rank": _mean(retrieval.get("median_rank", {})) if available else None,
                "mrr": _mean(retrieval.get("mrr", {})) if available else None,
                "std": macro_top10.get("std") if available else None,
                "ci_low": macro_top10.get("ci95_low") if available else None,
                "ci_high": macro_top10.get("ci95_high") if available else None,
            }
        )
    control_order = {name: index for index, name in enumerate(_CONTROL_NAMES.values())}
    return sorted(
        rows,
        key=lambda row: (row["vocabulary_size"], control_order[row["control"]]),
    )


def _csv_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else value


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: _csv_value(row.get(field)) for field in fields} for row in rows
        )


def _manifest_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _source(path: Path, project_root: Path, **metadata) -> dict:
    return {
        **metadata,
        "path": _manifest_path(path, project_root),
        "sha256": file_sha256(path),
    }


def _check_identity(payload: dict, selector: str, path: Path) -> None:
    identity = payload.get("experiment", {})
    actual = "/".join(
        str(identity.get(key, ""))
        for key in ("dataset", "subject_scope", "experiment_id")
    )
    if actual != selector:
        raise ValueError(f"实验身份与 selector 不一致：{path}")


def export_paper_results(
    export_dir: str | Path = PROJECT_ROOT / "reports" / "exports",
    *,
    config_root: str | Path | None = None,
    output_root: str | Path | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> dict:
    """读取正式 canonical JSON 并写出 Figure 2--4 的 CSV 与 manifest。"""
    export_dir = Path(export_dir)
    project_root = Path(project_root)
    output_root = Path(output_root) if output_root is not None else project_root / "outputs"
    fig2_rows = []
    fig3_rows = []
    fig4_rows = []
    sources = []
    runs = []
    included_selectors = []
    available_splits = set()

    for dataset, model, selector in paper_selectors():
        record = resolve_selector(selector, config_root=config_root)
        run_path = run_directory(record["raw_config"], output_root=output_root)
        included_selectors.append(selector)

        manifest_path = run_path / "run_manifest.json"
        training_path = run_path / "training_summary.json"
        manifest = _read_json(manifest_path)
        training = _read_json(training_path)
        _check_identity(manifest, selector, manifest_path)
        _check_identity(training, selector, training_path)
        sources.extend(
            (
                _source(
                    manifest_path,
                    project_root,
                    dataset=dataset,
                    model=model,
                    kind="run_manifest",
                ),
                _source(
                    training_path,
                    project_root,
                    dataset=dataset,
                    model=model,
                    kind="training_summary",
                ),
            )
        )

        checkpoint_sha = training.get("checkpoint", {}).get("best", {}).get("sha256")
        runs.append(
            {
                "dataset": dataset,
                "model": model,
                "selector": selector,
                "scientific_config_sha": manifest.get("scientific_config_sha256"),
                "checkpoint_sha": checkpoint_sha,
            }
        )

        for split in ("val", "test"):
            evaluation_path = run_path / "evaluation" / f"{split}.json"
            if split == "val" and not evaluation_path.is_file():
                raise FileNotFoundError(f"缺少 validation canonical JSON：{evaluation_path}")
            if not evaluation_path.is_file():
                continue
            evaluation = _read_json(evaluation_path)
            _check_identity(evaluation, selector, evaluation_path)
            if evaluation.get("data", {}).get("split") != split:
                raise ValueError(f"评价 JSON 的 split 与路径不一致：{evaluation_path}")
            evaluation_checkpoint = evaluation.get("checkpoint", {}).get("sha256")
            if checkpoint_sha and evaluation_checkpoint != checkpoint_sha:
                raise ValueError(f"训练与评价 checkpoint SHA 不一致：{evaluation_path}")
            available_splits.add(split)
            fig2_rows.extend(_evaluation_rows(dataset, model, evaluation))
            fig3_rows.extend(_information_rows(dataset, model, evaluation))
            sources.append(
                _source(
                    evaluation_path,
                    project_root,
                    dataset=dataset,
                    model=model,
                    kind="evaluation",
                    split=split,
                )
            )

            audit_directory = "validation" if split == "val" else split
            audit_path = run_path / "audits" / audit_directory / "summary.json"
            if split == "val" and not audit_path.is_file():
                raise FileNotFoundError(f"缺少 validation audit JSON：{audit_path}")
            if not audit_path.is_file():
                continue
            audit = _read_json(audit_path)
            _check_identity(audit, selector, audit_path)
            if audit.get("query_set", {}).get("split") != split:
                raise ValueError(f"审计 JSON 的 split 与路径不一致：{audit_path}")
            audit_checkpoint = audit.get("checkpoint", {}).get("sha256")
            if checkpoint_sha and audit_checkpoint != checkpoint_sha:
                raise ValueError(f"训练与审计 checkpoint SHA 不一致：{audit_path}")
            fig4_rows.extend(_audit_rows(dataset, model, audit))
            sources.append(
                _source(
                    audit_path,
                    project_root,
                    dataset=dataset,
                    model=model,
                    kind="audit_summary",
                    split=split,
                )
            )

    dataset_order = {name: index for index, name in enumerate(PAPER_DATASETS)}
    model_order = {name: index for index, name in enumerate(_MODEL_RUNS)}
    split_order = {"val": 0, "test": 1}
    control_order = {"clean": 0, "temporal": 1, "donor": 2, "structure_only": 3}
    for rows in (fig2_rows, fig3_rows, fig4_rows):
        rows.sort(
            key=lambda row: (
                dataset_order[row["dataset"]],
                split_order.get(row["split"], 99),
                model_order[row["model"]],
                row["vocabulary_size"],
                row.get("reference", ""),
                control_order.get(row.get("control"), -1),
            )
        )

    _write_csv(export_dir / "fig2_decoding_performance.csv", FIG2_FIELDS, fig2_rows)
    _write_csv(export_dir / "fig3_information.csv", FIG3_FIELDS, fig3_rows)
    _write_csv(export_dir / "fig4_audit.csv", FIG4_FIELDS, fig4_rows)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "datasets": list(PAPER_DATASETS),
        "included_selectors": included_selectors,
        "excluded_datasets": ["LibriBrain100"],
        "available_splits": sorted(available_splits),
        "runs": runs,
        "sources": sources,
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    manifest = export_paper_results()
    print(
        "论文数据导出完成："
        f"{', '.join(manifest['available_splits'])} -> reports/exports"
    )


if __name__ == "__main__":
    main()
