"""Canonical 运行的统一结果 JSON 结构。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from braindecoding.experiment import (
    evaluation_output_path,
    experiment_identity,
    file_sha256,
)
from braindecoding.evaluation.ovmi import fixed_vocabulary_ovmi_metrics
from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval_metrics


SCHEMA_VERSION = 1
PRIMARY_VOCABULARY_SIZES = (20, 50, 100, 150)
AUDIT_CONTROLS = (
    "clean",
    "temporal_shift",
    "donor_swap",
    "structure_only",
    "donor_following",
)


def canonical_retrieval(metrics):
    """把现有公共 retrieval 字段无数值变换地映射到稳定名称。"""
    metrics = metrics or {}
    return {
        "top1": metrics.get("micro_recall_at_1"),
        "top10": metrics.get("micro_recall_at_10"),
        "macro_top1": metrics.get("macro_recall_at_1"),
        "macro_top10": metrics.get("macro_recall_at_10"),
        "median_rank": metrics.get("median_rank"),
        "mrr": metrics.get("mean_reciprocal_rank"),
    }


def canonical_vocabulary_result(
    candidate_count,
    *,
    manifest_sha256=None,
    retrieval_metrics=None,
    ovmi_story=None,
    ovmi_domain=None,
    coverage=None,
):
    """构造单个冻结候选词表的公共评价结果块。"""
    candidate_count = int(candidate_count)
    metrics = retrieval_metrics or {}
    missing_words = list(metrics.get("missing_vocabulary_words", ()))
    supported_count = metrics.get("observed_vocabulary_size")
    if supported_count is None and retrieval_metrics is not None:
        supported_count = candidate_count - len(missing_words)
    support = {
        "supported_candidate_count": (
            int(supported_count) if supported_count is not None else None
        ),
        "missing_candidate_count": (
            len(missing_words) if retrieval_metrics is not None else None
        ),
        "missing_words": missing_words,
    }

    if missing_words:
        story = {
            "available": False,
            "reason": "missing_true_class_support",
        }
    elif ovmi_story is None:
        story = {"status": "not_run"}
    else:
        story = copy.deepcopy(ovmi_story)
        if story.get("reason") in {
            "full_ovmi_requires_true_samples_for_every_word",
            "missing_true_samples_in_frozen_vocabulary",
        }:
            story = {
                "available": False,
                "reason": "missing_true_class_support",
            }
    domain = (
        copy.deepcopy(ovmi_domain)
        if ovmi_domain is not None
        else {
            "available": False,
            "reason": "domain_reference_not_frozen",
        }
    )
    if coverage is None and story.get("available"):
        coverage = story.get("coverage")
    result = {
        "candidate_count": candidate_count,
        "manifest_sha256": manifest_sha256,
        "support": support,
        "coverage": coverage,
        "retrieval": canonical_retrieval(metrics),
        "ovmi": {"story": story, "domain": domain},
    }
    if retrieval_metrics is None:
        result["status"] = "not_run"
    return result


def load_vocabulary_assets(directory):
    """从调用方指定目录读取四个冻结词表和独立 story reference。"""
    directory = Path(directory)
    manifests = {}
    for size in PRIMARY_VOCABULARY_SIZES:
        path = directory / f"vocabulary_N{size}.json"
        if not path.is_file():
            raise FileNotFoundError(f"找不到冻结候选词表：{path}")
        manifests[size] = json.loads(path.read_text(encoding="utf-8"))
    reference_path = directory / "story_reference.json"
    if not reference_path.is_file():
        raise FileNotFoundError(f"找不到冻结 story reference：{reference_path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    return manifests, reference


def evaluate_vocabulary_manifests(
    predictions,
    targets,
    words,
    vocabulary_manifests,
    story_reference,
    *,
    language,
):
    """用现有公共数学一次生成四个冻结词表的 canonical 结果块。"""
    results = {}
    for size in PRIMARY_VOCABULARY_SIZES:
        manifest = vocabulary_manifests.get(size)
        if manifest is None:
            raise ValueError(f"缺少 N{size} 冻结候选词表 manifest。")
        vocabulary = manifest["vocabulary"]
        if len(vocabulary) != size:
            raise ValueError(f"N{size} manifest 的实际词数不一致。")
        metrics = fixed_vocabulary_retrieval_metrics(
            predictions,
            targets,
            words,
            vocabulary,
            top_ks=(1, 10),
            vocabulary_name=f"frozen_N{size}",
        )
        ovmi_story = fixed_vocabulary_ovmi_metrics(
            predictions,
            targets,
            words,
            vocabulary,
            {
                "enabled": True,
                "method": "full",
                "reference": story_reference,
                "language": language,
            },
        )
        results[size] = canonical_vocabulary_result(
            size,
            manifest_sha256=manifest.get("manifest_sha256"),
            retrieval_metrics=metrics,
            ovmi_story=ovmi_story,
        )
    return results


def build_evaluation_result(
    config,
    *,
    split,
    query_count,
    event_table_sha256,
    subjects,
    checkpoint,
    vocabulary_results,
    status="evaluation_completed",
):
    """构造跨数据集一致的 canonical evaluation JSON。"""
    if split not in {"val", "test"}:
        raise ValueError("评价 split 只能是 val 或 test。")
    supplied = {int(size): copy.deepcopy(value) for size, value in vocabulary_results.items()}
    vocabularies = {}
    for size in PRIMARY_VOCABULARY_SIZES:
        block = supplied.get(size, canonical_vocabulary_result(size))
        if int(block.get("candidate_count", -1)) != size:
            raise ValueError(f"词表结果块与 N{size} 的候选数量不一致。")
        vocabularies[str(size)] = block
    unknown = sorted(set(supplied) - set(PRIMARY_VOCABULARY_SIZES))
    if unknown:
        raise ValueError(f"canonical 主评价不支持词表规模：{unknown}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "experiment": experiment_identity(config),
        "data": {
            "split": split,
            "query_count": int(query_count),
            "event_table_sha256": event_table_sha256,
            "subjects": [str(subject) for subject in subjects],
        },
        "checkpoint": copy.deepcopy(checkpoint),
        "vocabularies": vocabularies,
    }


def canonical_evaluation_from_legacy(
    config,
    legacy_summary,
    *,
    event_table_path,
    subjects,
    checkpoint_path,
    checkpoint,
    vocabulary_manifest_sha256=None,
    vocabulary_results=None,
):
    """把未改数学的现有评价结果封装成 canonical 文件结构。"""
    metrics = legacy_summary["metrics"]
    candidate_count = int(metrics.get("vocabulary_size", 0))
    if candidate_count not in PRIMARY_VOCABULARY_SIZES:
        raise ValueError(f"评价候选词表规模不属于冻结主集合：{candidate_count}")
    if vocabulary_results is None:
        block = canonical_vocabulary_result(
            candidate_count,
            manifest_sha256=vocabulary_manifest_sha256,
            retrieval_metrics=metrics,
            ovmi_story=metrics.get("ovmi"),
        )
        vocabulary_results = {candidate_count: block}
    return build_evaluation_result(
        config,
        split=legacy_summary["split"],
        query_count=legacy_summary["query_rows"],
        event_table_sha256=file_sha256(event_table_path),
        subjects=subjects,
        checkpoint={
            "path": Path(checkpoint_path).name,
            "sha256": file_sha256(checkpoint_path),
            "epoch": (
                int(checkpoint["epoch"])
                if checkpoint.get("epoch") is not None
                else None
            ),
            "update": (
                int(checkpoint["optimizer_updates"])
                if checkpoint.get("optimizer_updates") is not None
                else None
            ),
        },
        vocabulary_results=vocabulary_results,
        status=legacy_summary["status"],
    )


def write_evaluation_result(config, result, *, smoke=False):
    """按 canonical/legacy 路径规则保存一个明确 split 的评价结果。"""
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("canonical evaluation 缺少受支持的 schema_version。")
    split = result.get("data", {}).get("split")
    path = evaluation_output_path(config, split, smoke=smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _best_update(legacy_summary):
    best_epoch = legacy_summary.get("best_epoch")
    for record in legacy_summary.get("history", ()):  # 保留原 epoch/update 语义。
        if record.get("epoch") == best_epoch:
            value = record.get("optimizer_updates")
            return int(value) if value is not None else None
    return None


def _checkpoint_result(output_dir, name):
    path = Path(output_dir) / name
    return {"path": name, "sha256": file_sha256(path)}


def build_training_summary(config, legacy_summary):
    """在不改训练循环的前提下封装 update/epoch 两种预算。"""
    training_config = config.get("training", {})
    target_updates = legacy_summary.get("target_updates")
    if target_updates is None:
        target_updates = training_config.get("max_updates")
    if target_updates is not None:
        budget = {"type": "updates", "value": int(target_updates)}
        completed_updates = legacy_summary.get("optimizer_updates")
        completed_updates = (
            int(completed_updates) if completed_updates is not None else None
        )
        best_update = _best_update(legacy_summary)
    else:
        budget = {"type": "epochs", "value": int(training_config["epochs"])}
        completed_updates = None
        best_update = None
    history = legacy_summary.get("history", ())
    output_dir = training_config.get("output_dir", ".")
    best_epoch = legacy_summary.get("best_epoch")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment_identity(config),
        "status": legacy_summary.get("status"),
        "training": {
            "budget": budget,
            "completed_epochs": int(len(history)),
            "completed_updates": completed_updates,
        },
        "selection": {
            "split": "val",
            "metric": legacy_summary.get("selection_metric"),
            "best_value": legacy_summary.get("best_score"),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_update": best_update,
        },
        "checkpoint": {
            "best": _checkpoint_result(output_dir, "best.pt"),
            "last": _checkpoint_result(output_dir, "last.pt"),
        },
        "test_status": legacy_summary.get("test_status", "locked_not_evaluated"),
        "details": copy.deepcopy(legacy_summary),
    }


def build_audit_summary(config, *, checkpoint, query_set, controls):
    """构造固定 control 名称、未运行项显式标记的 validation audit 摘要。"""
    if query_set.get("split") != "val":
        raise ValueError("canonical audit summary 本轮只允许 validation query set。")
    unknown = sorted(set(controls) - set(AUDIT_CONTROLS))
    if unknown:
        raise ValueError(f"未知 audit control：{unknown}")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment_identity(config),
        "checkpoint": copy.deepcopy(checkpoint),
        "query_set": copy.deepcopy(query_set),
        "controls": {
            name: copy.deepcopy(controls.get(name, {"status": "not_run"}))
            for name in AUDIT_CONTROLS
        },
    }


def write_audit_summary(config, result):
    """把公共审计摘要写入当前 canonical run 的 validation 目录。"""
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("canonical audit summary 缺少受支持的 schema_version。")
    path = (
        Path(config["training"]["output_dir"])
        / "audits"
        / "validation"
        / "summary.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
