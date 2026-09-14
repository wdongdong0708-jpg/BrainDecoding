"""按 exact experiment selector 运行 validation-only canonical audit。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from braindecoding.audit.controls import (
    DONOR_SEEDS,
    build_control_assets,
    file_sha256,
    json_bytes,
    load_control_asset,
    validate_payload_sha256,
    write_control_assets,
)
from braindecoding.audit.evaluate import (
    aggregate_donor_results,
    collect_brain_features,
    evaluate_control,
    predict_from_cached_features,
    predict_structure_only_from_cached_features,
    save_audit_result,
    save_feature_cache,
    verify_cached_clean_predictions,
    verify_cached_donor_features,
)
from braindecoding.catalog import resolve_selector
from braindecoding.config import PROJECT_ROOT
from braindecoding.experiment import experiment_identity, scientific_config_sha256
from braindecoding.models import build_brain_embedding_model
from braindecoding.protocols import validate_manifest_sha256
from braindecoding.results import build_audit_summary, write_audit_summary


MANIFEST_ROOT = PROJECT_ROOT / "experiments" / "manifests"
DETAIL_ROOT = PROJECT_ROOT / "reports" / "audits" / "validation_controls"
PRIMARY_VOCABULARY_SIZES = (20, 50, 100, 150)
DATASET_LABELS = {
    "chineseeeg2_littleprince": "ChineseEEG2",
    "smn4lang": "SMN4Lang",
    "libribrain100": "LibriBrain100",
    "pallier2025": "Pallier2025",
}
DATASET_LANGUAGES = {
    "chineseeeg2_littleprince": "zh",
    "smn4lang": "zh",
    "libribrain100": "en",
    "pallier2025": "fr",
}
MANIFEST_DIRECTORIES = {
    "chineseeeg2_littleprince": "chineseeeg2",
    "smn4lang": "smn4lang",
    "pallier2025": "pallier2025",
}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_condition_from_config(config: dict) -> str:
    """只从实际模型配置判定 word 或 grouped neural_context。"""
    model = config.get("model", {})
    if not bool(model.get("use_transformer", False)):
        return "word"
    if model.get("context_mode") == "grouped":
        return "neural_context"
    raise ValueError(
        "audit 只支持 use_transformer=false 的 word，或 grouped neural_context。"
    )


def _task_adapter(dataset_id: str) -> dict:
    """返回四个数据集现有训练入口的最小函数适配。"""
    if dataset_id == "chineseeeg2_littleprince":
        from braindecoding.data import chineseeeg2 as data_module
        from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import (
            evaluate as evaluation,
        )
        from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import (
            train as training,
        )

        return {
            "build_dataset": training.构建数据集,
            "checkpoint_task": "word_decoding/ChineseEEG2_LittlePrince",
            "data_module": data_module,
            "load_config": training.载入配置,
            "load_event_table": data_module.载入事件表,
            "make_loader": training.make_loader,
            "signal_key": "eeg",
            "task_checkpoint_validator": evaluation.验证检查点合同,
        }
    if dataset_id == "smn4lang":
        from braindecoding.data import smn4lang as data_module
        from braindecoding.tasks.word_decoding.smn4lang import train as training

        return {
            "build_dataset": training.build_dataset,
            "checkpoint_task": "word_decoding/SMN4Lang",
            "data_module": data_module,
            "load_config": training.load_config,
            "load_event_table": data_module.load_event_table,
            "make_loader": training.make_loader,
            "signal_key": "meg",
        }
    if dataset_id == "libribrain100":
        from braindecoding.data import libribrain as data_module
        from braindecoding.tasks.word_decoding.libribrain100 import train as training

        return {
            "build_dataset": training.build_dataset,
            "checkpoint_task": "word_decoding/LibriBrain100",
            "data_module": data_module,
            "load_config": training.load_config,
            "load_event_table": data_module.load_event_table,
            "make_loader": training.make_loader,
            "signal_key": "meg",
        }
    if dataset_id == "pallier2025":
        from braindecoding.data import pallier2025 as data_module
        from braindecoding.tasks.word_decoding.pallier2025 import train as training

        return {
            "build_dataset": training.build_dataset,
            "checkpoint_task": "word_decoding/Pallier2025",
            "data_module": data_module,
            "group_column": training.loader_group_column,
            "load_config": training.load_config,
            "load_event_table": data_module.load_event_table,
            "make_loader": training.make_loader,
            "signal_key": "meg",
        }
    raise ValueError(f"audit 尚不支持数据集：{dataset_id}")


def _load_protocol_assets(dataset_id: str) -> dict:
    """读取已冻结资产；Libri 缺失的 N 和 story 不会被自动构造。"""
    language = DATASET_LANGUAGES[dataset_id]
    if dataset_id == "libribrain100":
        from braindecoding.data.libribrain import LIBRIBRAIN100_50_WORD_VOCABULARY

        vocabulary = list(LIBRIBRAIN100_50_WORD_VOCABULARY)
        content_sha = hashlib.sha256(
            json.dumps(
                vocabulary, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        missing = {
            size: {
                "candidate_count": size,
                "reason": "vocabulary_manifest_not_frozen",
                "status": "not_run",
            }
            for size in PRIMARY_VOCABULARY_SIZES
            if size != 50
        }
        return {
            "language": language,
            "story_reference": None,
            "support": None,
            "vocabularies": {
                50: {
                    "dataset": "libribrain100",
                    "manifest_sha256": None,
                    "source": "frozen_in_source_libribrain100_50",
                    "vocabulary": vocabulary,
                    "vocabulary_content_sha256": content_sha,
                    "vocabulary_size": 50,
                }
            },
            "vocabulary_statuses": missing,
        }

    directory = MANIFEST_ROOT / MANIFEST_DIRECTORIES[dataset_id]
    vocabularies = {}
    for size in PRIMARY_VOCABULARY_SIZES:
        path = directory / f"vocabulary_N{size}.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少已冻结 N{size} 候选词表：{path}")
        manifest = _load_json(path)
        validate_manifest_sha256(manifest)
        vocabularies[size] = manifest
    story_path = directory / "story_reference.json"
    if not story_path.is_file():
        raise FileNotFoundError(f"缺少 story reference：{story_path}")
    support_path = directory / "ovmi_support.json"
    if not support_path.is_file():
        raise FileNotFoundError(f"缺少 OVMI support manifest：{support_path}")
    support = _load_json(support_path)
    validate_manifest_sha256(support)
    return {
        "language": language,
        "story_reference": _load_json(story_path),
        "support": support,
        "vocabularies": vocabularies,
        "vocabulary_statuses": {},
    }


def _restore_control_columns(table: pd.DataFrame, config: dict) -> pd.DataFrame:
    """仅在内存恢复控制映射所需英文字段，不改变 canonical CSV。"""
    result = table.copy()
    aliases = {
        "event_id": "事件编号",
        "subject_id": "受试者",
        "recording_id": "记录编号",
        "normalized_word": "标准词",
        "sentence_uid": "上下文编号",
        "split": "数据划分",
        "is_trainable": "是否可训练",
    }
    for target, source in aliases.items():
        if target not in result and source in result:
            result[target] = result[source]
    if "window_start_seconds" not in result:
        if "开始时间" not in result:
            raise ValueError("事件表缺少控制窗口起点。")
        offset = float(config["dataset"].get("window_start_offset_seconds", 0.0))
        result["window_start_seconds"] = pd.to_numeric(
            result["开始时间"], errors="raise"
        ) + offset
    if "window_stop_seconds" not in result:
        result["window_stop_seconds"] = (
            pd.to_numeric(result["window_start_seconds"], errors="raise")
            + float(config["dataset"]["window_seconds"])
        )
    if "window_complete" not in result:
        result["window_complete"] = result["is_trainable"]
    return result


def _load_exact_config(selector: str) -> tuple[dict, dict, dict]:
    """解析 selector 后通过对应 task loader 展开同一份 config。"""
    record = resolve_selector(selector)
    dataset_id = record["identity"]["dataset"]
    adapter = _task_adapter(dataset_id)
    config = adapter["load_config"](record["config_path"])
    if experiment_identity(config) != record["identity"]:
        raise ValueError("selector 身份与展开后的 experiment identity 不一致。")
    return record, config, adapter


def _validation_table(config: dict, adapter: dict) -> pd.DataFrame:
    """只返回 exact config 的可训练 validation 事件。"""
    path = Path(config["cache"]["event_table"])
    if not path.is_file():
        raise FileNotFoundError(f"canonical event table 不存在：{path}")
    table = adapter["load_event_table"](
        path,
        split="val",
        subjects=config["dataset"].get("subjects"),
        trainable_only=True,
    )
    table = _restore_control_columns(table, config)
    if not table["split"].eq("val").all():
        raise AssertionError("audit validation table 意外含非 val 事件。")
    return table


def _group_column(config: dict, adapter: dict) -> str:
    resolver = adapter.get("group_column")
    return resolver(config) if resolver is not None else "sentence_uid"


def validate_runtime_contexts(dataset, group_column: str, max_context_words: int) -> dict:
    """验证 Transformer 实际分组没有跨受试者、记录或 canonical context。"""
    table = dataset.table.reset_index(drop=True)
    required = {group_column, "subject_id", "recording_id", "sentence_uid"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"runtime context 验证缺少字段：{missing}")
    groups = table.groupby(group_column, sort=False, dropna=False)
    subject_counts = groups["subject_id"].nunique()
    recording_counts = groups["recording_id"].nunique()
    canonical_counts = groups["sentence_uid"].nunique()
    sizes = groups.size()
    result = {
        "group_column": str(group_column),
        "group_count": int(len(sizes)),
        "mean_words": float(sizes.mean()),
        "median_words": float(sizes.median()),
        "maximum_words": int(sizes.max()),
        "cross_subject_groups": int(subject_counts.gt(1).sum()),
        "cross_recording_groups": int(recording_counts.gt(1).sum()),
        "cross_canonical_context_groups": int(canonical_counts.gt(1).sum()),
        "groups_over_max_context_words": int(
            sizes.gt(int(max_context_words)).sum()
        ),
        "record_order_strictly_increasing": True,
    }
    if any(
        result[key]
        for key in (
            "cross_subject_groups",
            "cross_recording_groups",
            "cross_canonical_context_groups",
            "groups_over_max_context_words",
        )
    ):
        raise ValueError(f"runtime context 合同失败：{result}")
    if "记录内序号" in table:
        for context_uid, rows in groups:
            values = pd.to_numeric(
                rows["记录内序号"], errors="coerce"
            ).to_numpy()
            if np.isnan(values).any() or (np.diff(values) <= 0).any():
                raise ValueError(
                    f"runtime context 记录内序号未严格递增：{context_uid}"
                )
    group_values = table[group_column].astype(str).tolist()
    expected_map = {
        value: index for index, value in enumerate(dict.fromkeys(group_values))
    }
    expected_indices = np.asarray(
        [expected_map[value] for value in group_values], dtype=np.int64
    )
    if not np.array_equal(expected_indices, np.asarray(dataset.sentence_indices)):
        raise ValueError("Dataset sentence_index 未使用 loader 的 runtime context 分组。")
    return result


def _build_loader_bundle(selector: str, device, *, load_model: bool) -> dict:
    """构建 exact selector 的 validation loader，并按需载入其唯一 best.pt。"""
    record, config, adapter = _load_exact_config(selector)
    model_condition = model_condition_from_config(config)
    table = _validation_table(config, adapter)
    dataset = adapter["build_dataset"](config, table)
    group_column = _group_column(config, adapter)
    loader, _ = adapter["make_loader"](
        dataset,
        config["training"],
        shuffle=False,
        group_column=group_column,
    )
    runtime_context = None
    if model_condition == "neural_context":
        runtime_context = validate_runtime_contexts(
            dataset,
            group_column,
            int(config["dataset"].get("max_context_words", 128)),
        )
    bundle = {
        "adapter": adapter,
        "config": config,
        "dataset": dataset,
        "event_path": Path(config["cache"]["event_table"]),
        "group_column": group_column,
        "identity": record["identity"],
        "loader": loader,
        "model_condition": model_condition,
        "runtime_context": runtime_context,
        "signal_key": adapter["signal_key"],
        "table": table,
    }
    if not load_model:
        return bundle
    checkpoint_path = Path(config["training"]["output_dir"]) / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"exact experiment checkpoint 不存在：{checkpoint_path}\n"
            f"experiment not auditable: {selector}"
        )
    from braindecoding.training.runtime import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    _validate_checkpoint(checkpoint, bundle)
    model = build_brain_embedding_model(
        dataset.channel_count,
        dataset.channel_positions,
        dataset.subject_count,
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    bundle.update(
        {
            "checkpoint": checkpoint,
            "checkpoint_path": checkpoint_path,
            "model": model,
        }
    )
    return bundle


def _validate_checkpoint(checkpoint: dict, bundle: dict) -> None:
    """拒绝 selector config 与 checkpoint 模型/数据身份不一致。"""
    adapter = bundle["adapter"]
    config = bundle["config"]
    dataset = bundle["dataset"]
    if checkpoint.get("task") != adapter["checkpoint_task"]:
        raise ValueError("checkpoint task 与 exact selector 不一致。")
    if checkpoint.get("model_config") != config["model"]:
        raise ValueError("checkpoint model_config 与 exact selector 不一致。")
    checkpoint_text = checkpoint.get("text_embedding_config")
    if checkpoint_text is not None and checkpoint_text != config["text_embedding"]:
        raise ValueError("checkpoint text embedding 合同与 exact selector 不一致。")
    if checkpoint.get("loss_config") is not None and (
        checkpoint["loss_config"] != config.get("loss")
    ):
        raise ValueError("checkpoint loss 合同与 exact selector 不一致。")
    if checkpoint.get("provenance") is not None and (
        checkpoint["provenance"] != config.get("provenance")
    ):
        raise ValueError("checkpoint provenance 与 exact selector 不一致。")
    if model_condition_from_config(
        {"model": checkpoint["model_config"]}
    ) != bundle["model_condition"]:
        raise ValueError("checkpoint 的 word/context 条件与 exact selector 不一致。")
    channels = tuple(checkpoint.get("channel_names", ()))
    if channels and channels != tuple(dataset.channel_names):
        raise ValueError("checkpoint 通道顺序与 exact selector 数据不一致。")
    positions = np.asarray(checkpoint.get("channel_positions", ()))
    expected_positions = np.asarray(dataset.channel_positions)
    if positions.size and (
        positions.shape != expected_positions.shape
        or not np.allclose(positions, expected_positions, rtol=0.0, atol=1e-7)
    ):
        raise ValueError("checkpoint 传感器位置与 exact selector 数据不一致。")
    contract = checkpoint.get("dataset_contract", {})
    configured_subjects = tuple(config["dataset"].get("subjects", ()))
    checkpoint_subjects = tuple(
        contract.get("subject_order", contract.get("subjects", ()))
    )
    if configured_subjects and checkpoint_subjects and (
        configured_subjects != checkpoint_subjects
    ):
        raise ValueError("checkpoint 受试者顺序与 exact selector 不一致。")
    for field in (
        "window_start_offset_seconds",
        "window_seconds",
        "eligibility_window_seconds",
        "baseline_seconds",
        "clamp",
        "target_sampling_rate_hz",
        "context_grouping",
        "max_context_words",
        "training_vocabulary_policy",
        "evaluation_vocabulary_policy",
    ):
        expected = config["dataset"].get(field)
        if field in contract and expected != contract[field]:
            raise ValueError(f"checkpoint 数据合同与 exact selector 不一致：{field}")
    if bundle["identity"]["dataset"] == "pallier2025":
        expected_runtime = config["dataset"].get("runtime_context_grouping")
        if bundle["model_condition"] == "neural_context" and (
            contract.get("runtime_context_grouping") != expected_runtime
        ):
            raise ValueError(
                "Pallier2025 checkpoint 未使用当前 recording-local runtime context。"
            )


def _mapping_directory(dataset_id: str) -> Path:
    stable_dataset_key = MANIFEST_DIRECTORIES.get(dataset_id, dataset_id)
    return DETAIL_ROOT / stable_dataset_key / "validation" / "mappings"


def _detail_directory(identity: dict) -> Path:
    return (
        DETAIL_ROOT
        / identity["dataset"]
        / "validation"
        / "experiments"
        / identity["subject_scope"]
        / identity["experiment_id"]
        / f"seed-{identity['seed']:03d}"
    )


def _write_or_validate_mappings(assets: dict[str, dict], directory: Path) -> None:
    """映射是 dataset-level 资产；存在时必须逐字节相同。"""
    for name, payload in sorted(assets.items()):
        path = directory / name
        expected = json_bytes(payload)
        if path.is_file() and path.read_bytes() != expected:
            raise ValueError(f"既有 dataset-level audit mapping 漂移：{path}")
    write_control_assets(assets, directory)


def _expected_mapping_names() -> tuple[str, ...]:
    return (
        "clean_identity_mapping.json",
        "core_audit_queries.json",
        "temporal_shift_mapping.json",
        *(f"donor_swap_seed{seed:02d}.json" for seed in DONOR_SEEDS),
    )


def _load_existing_mappings(
    directory: Path, *, dataset_label: str, event_table_sha256: str
) -> dict[str, dict] | None:
    """完整映射集存在时严格验证并复用；拒绝部分资产。"""
    names = _expected_mapping_names()
    existing = [name for name in names if (directory / name).is_file()]
    if not existing:
        return None
    if len(existing) != len(names):
        missing = sorted(set(names) - set(existing))
        raise ValueError(f"dataset-level audit mapping 不完整：{missing}")
    assets = {}
    for name in names:
        sha_field = (
            "query_manifest_sha256"
            if name == "core_audit_queries.json"
            else "mapping_sha256"
        )
        payload = load_control_asset(directory / name, sha_field)
        if payload.get("dataset") != dataset_label:
            raise ValueError(f"audit mapping 数据集身份不一致：{directory / name}")
        if payload.get("event_table_sha256") != event_table_sha256:
            raise ValueError(f"audit mapping 事件表 SHA 不一致：{directory / name}")
        assets[name] = payload
    if assets["clean_identity_mapping.json"].get("algorithm") != "identity":
        raise ValueError("clean mapping 算法合同漂移。")
    temporal_algorithm = assets["temporal_shift_mapping.json"].get("algorithm", {})
    if temporal_algorithm.get("uses_word_labels") is not False:
        raise ValueError("temporal mapping 不得使用词标签。")
    for seed in DONOR_SEEDS:
        payload = assets[f"donor_swap_seed{seed:02d}.json"]
        if payload.get("seed") != seed:
            raise ValueError(f"donor mapping seed 合同漂移：{seed}")
        algorithm = payload.get("algorithm", {})
        if (
            algorithm.get("random_generator") != "numpy.random.default_rng"
            or algorithm.get("candidate_sampling") != "uniform_per_target"
            or algorithm.get("replacement_across_targets") is not True
        ):
            raise ValueError(f"donor mapping 算法合同漂移：seed={seed}")
    return assets


def _build_mappings(bundle: dict) -> dict[str, dict]:
    identity = bundle["identity"]
    dataset_label = DATASET_LABELS[identity["dataset"]]
    event_sha = file_sha256(bundle["event_path"])
    directory = _mapping_directory(identity["dataset"])
    existing = _load_existing_mappings(
        directory,
        dataset_label=dataset_label,
        event_table_sha256=event_sha,
    )
    if existing is not None:
        return existing
    assets = build_control_assets(
        bundle["table"],
        dataset=dataset_label,
        event_table_sha256=event_sha,
        split="val",
        status="validation_only",
    )
    _write_or_validate_mappings(assets, directory)
    return assets


def _artifact_reference(path: Path, sha_field: str | None = None) -> dict:
    payload = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    if sha_field is not None:
        content = _load_json(path)
        validate_payload_sha256(content, sha_field)
        payload[sha_field] = content[sha_field]
    return payload


def _evaluate_predictions(predictions, encoded, query_ids, protocol, dataset_id):
    return evaluate_control(
        predictions,
        encoded,
        query_ids,
        protocol["vocabularies"],
        protocol["support"],
        protocol["story_reference"],
        dataset_name=dataset_id,
        language=protocol["language"],
        vocabulary_statuses=protocol["vocabulary_statuses"],
    )


def _saved_clean_comparison(config, encoded: dict, query_event_ids, metrics) -> dict:
    """只有 query event set 完全相同时才比较 canonical full-val 数值。"""
    if set(map(str, query_event_ids)) != set(map(str, encoded["event_ids"])):
        return {
            "available": False,
            "reason": "full_validation_vs_core_query_set",
        }
    output = Path(config["training"]["output_dir"])
    canonical_path = output / "evaluation" / "val.json"
    legacy_path = output / "evaluation_val.json"
    if canonical_path.is_file():
        saved = _load_json(canonical_path)
        if saved.get("data", {}).get("query_count") != len(encoded["event_ids"]):
            return {
                "available": False,
                "reason": "saved_evaluation_query_count_mismatch",
            }
        block = saved.get("vocabularies", {}).get("50", {}).get("retrieval")
        if not block:
            return {"available": False, "reason": "saved_N50_metrics_missing"}
        actual = metrics.get("N50")
        if actual is None or actual.get("status") == "not_run":
            return {"available": False, "reason": "audit_N50_not_run"}
        pairs = {
            "top1": "micro_recall_at_1",
            "top10": "micro_recall_at_10",
            "macro_top1": "macro_recall_at_1",
            "macro_top10": "macro_recall_at_10",
            "median_rank": "median_rank",
            "mrr": "mean_reciprocal_rank",
        }
        deltas = {
            key: float(actual[source]) - float(block[key])
            for key, source in pairs.items()
            if block.get(key) is not None
        }
        exact = all(value == 0.0 for value in deltas.values())
        absolute_tolerance = 1e-12
        numerically_equal = all(
            abs(value) <= absolute_tolerance for value in deltas.values()
        )
        if not numerically_equal:
            raise AssertionError(f"同 query set 的 clean metrics 不一致：{deltas}")
        return {
            "available": True,
            "exactly_equal": exact,
            "numerically_equal": True,
            "absolute_tolerance": absolute_tolerance,
            "maximum_absolute_delta": max(map(abs, deltas.values()), default=0.0),
            "field_deltas": deltas,
            "saved_evaluation": str(canonical_path.resolve()),
        }
    if legacy_path.is_file():
        return {
            "available": False,
            "reason": "legacy_evaluation_has_no_query_event_ids",
            "saved_evaluation": str(legacy_path.resolve()),
        }
    return {"available": False, "reason": "saved_evaluation_val_missing"}


def _retrieval_differences(clean, temporal, donor_aggregate, structure) -> dict:
    """输出 raw-fraction 描述性差值，不宣称可加性或纯神经信息。"""
    fields = {
        "top1": "micro_recall_at_1",
        "top10": "micro_recall_at_10",
        "macro_top1": "macro_recall_at_1",
        "macro_top10": "macro_recall_at_10",
        "median_rank": "median_rank",
        "mrr": "mean_reciprocal_rank",
    }
    result = {}
    for size, clean_block in clean.items():
        if clean_block.get("status") == "not_run":
            continue
        current = {
            "unit": "raw_fraction_for_recall_and_mrr; rank_for_median_rank",
        }
        temporal_block = temporal.get(size, {})
        if temporal_block.get("status") != "not_run":
            current["clean_minus_temporal"] = {
                name: float(clean_block[field]) - float(temporal_block[field])
                for name, field in fields.items()
            }
        donor_block = donor_aggregate.get(size, {})
        if donor_block.get("status") == "completed":
            current["clean_minus_donor_mean"] = {
                name: float(clean_block[field])
                - float(donor_block["retrieval"][name]["mean"])
                for name, field in fields.items()
            }
        structure_block = structure.get(size, {}) if structure else {}
        if structure_block.get("status") != "not_run" and structure_block:
            current["clean_minus_structure_only"] = {
                name: float(clean_block[field]) - float(structure_block[field])
                for name, field in fields.items()
            }
        result[size] = current
    return result


def _run_full_audit(bundle: dict, mappings: dict, protocol: dict, device) -> dict:
    config = bundle["config"]
    identity = bundle["identity"]
    model = bundle["model"]
    checkpoint_path = bundle["checkpoint_path"]
    checkpoint_sha = file_sha256(checkpoint_path)
    core = mappings["core_audit_queries.json"]
    query_ids = core["event_ids"]
    amp = config["training"].get("amp", True)
    encoded = collect_brain_features(
        model,
        bundle["loader"],
        device,
        signal_key=bundle["signal_key"],
        amp=amp,
        collect_direct_predictions=True,
    )
    clean_prediction_check = verify_cached_clean_predictions(
        model,
        encoded,
        mappings["clean_identity_mapping.json"],
        query_ids,
        model_condition=bundle["model_condition"],
        device=device,
        amp=amp,
    )
    detail_dir = _detail_directory(identity)
    feature_cache = save_feature_cache(
        detail_dir / "feature_cache",
        encoded,
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "event_table_sha256": file_sha256(bundle["event_path"]),
            "experiment": identity,
            "model_condition": bundle["model_condition"],
            "split": "val",
            "test_neural_data_opened": False,
        },
    )
    donor_feature_check = verify_cached_donor_features(
        model,
        bundle["dataset"],
        encoded,
        mappings["donor_swap_seed00.json"],
        device,
        signal_key=bundle["signal_key"],
        amp=amp,
    )

    results = {}
    artifacts = {}
    for control_name, mapping_name in (
        ("clean", "clean_identity_mapping.json"),
        ("temporal_shift", "temporal_shift_mapping.json"),
    ):
        prediction, mapping_audit = predict_from_cached_features(
            model,
            encoded,
            mappings[mapping_name],
            query_ids,
            model_condition=bundle["model_condition"],
            device=device,
            amp=amp,
        )
        metrics = _evaluate_predictions(
            prediction, encoded, query_ids, protocol, identity["dataset"]
        )
        payload = {
            "checkpoint_sha256": checkpoint_sha,
            "control_condition": control_name,
            "mapping_audit": mapping_audit,
            "mapping_sha256": mappings[mapping_name]["mapping_sha256"],
            "metrics": metrics,
            "query_manifest_sha256": core["query_manifest_sha256"],
            "split": "val",
            "test_model_evaluation": "not_run",
            "test_predictions_generated": False,
        }
        path = save_audit_result(detail_dir / f"{control_name}.json", payload)
        results[control_name] = metrics
        artifacts[control_name] = _artifact_reference(path, "result_sha256")

    donor_per_seed = {}
    donor_artifacts = {}
    for seed in DONOR_SEEDS:
        name = f"donor_swap_seed{seed:02d}"
        mapping = mappings[f"{name}.json"]
        prediction, mapping_audit = predict_from_cached_features(
            model,
            encoded,
            mapping,
            query_ids,
            model_condition=bundle["model_condition"],
            device=device,
            amp=amp,
        )
        metrics = _evaluate_predictions(
            prediction, encoded, query_ids, protocol, identity["dataset"]
        )
        path = save_audit_result(
            detail_dir / f"{name}.json",
            {
                "checkpoint_sha256": checkpoint_sha,
                "control_condition": "donor_swap",
                "mapping_audit": mapping_audit,
                "mapping_sha256": mapping["mapping_sha256"],
                "metrics": metrics,
                "query_manifest_sha256": core["query_manifest_sha256"],
                "seed": seed,
                "split": "val",
                "test_model_evaluation": "not_run",
                "test_predictions_generated": False,
            },
        )
        donor_per_seed[f"seed-{seed:02d}"] = metrics
        donor_artifacts[f"seed-{seed:02d}"] = _artifact_reference(
            path, "result_sha256"
        )
    donor_aggregate = aggregate_donor_results(donor_per_seed)

    if bundle["model_condition"] == "neural_context":
        structure_prediction, structure_audit = (
            predict_structure_only_from_cached_features(
                model,
                encoded,
                model_condition=bundle["model_condition"],
                device=device,
                amp=amp,
            )
        )
        structure_metrics = _evaluate_predictions(
            structure_prediction, encoded, query_ids, protocol, identity["dataset"]
        )
        structure_payload = {
            "checkpoint_sha256": checkpoint_sha,
            "control_condition": "structure_only",
            "definition": "no_neural_feature_structural_inference_control",
            "implementation_audit": structure_audit,
            "metrics": structure_metrics,
            "query_manifest_sha256": core["query_manifest_sha256"],
            "split": "val",
            "test_model_evaluation": "not_run",
            "test_predictions_generated": False,
        }
        structure_path = save_audit_result(
            detail_dir / "structure_only.json", structure_payload
        )
        artifacts["structure_only"] = _artifact_reference(
            structure_path, "result_sha256"
        )
        structure_control = {
            "status": "completed",
            "definition": structure_payload["definition"],
            "results_by_vocabulary": structure_metrics,
        }
    else:
        structure_metrics = None
        structure_control = {
            "status": "not_applicable",
            "reason": "no_context_transformer",
        }

    controls = {
        "clean": {
            "status": "completed",
            "definition": "identity_neural_feature_mapping",
            "results_by_vocabulary": results["clean"],
        },
        "temporal_shift": {
            "status": "completed",
            "definition": "target_local_neural_substitution_control",
            "mapping_sha256": mappings["temporal_shift_mapping.json"][
                "mapping_sha256"
            ],
            "results_by_vocabulary": results["temporal_shift"],
        },
        "donor_swap": {
            "status": "completed",
            "definition": "target_local_neural_substitution_control",
            "seeds": list(DONOR_SEEDS),
            "mapping_sha256": {
                f"seed-{seed:02d}": mappings[f"donor_swap_seed{seed:02d}.json"][
                    "mapping_sha256"
                ]
                for seed in DONOR_SEEDS
            },
            "per_seed_results": donor_per_seed,
            "aggregate": donor_aggregate,
        },
        "structure_only": structure_control,
        "donor_following": {"status": "not_run"},
    }
    query_path = _mapping_directory(identity["dataset"]) / "core_audit_queries.json"
    summary = build_audit_summary(
        config,
        checkpoint={
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "epoch": bundle["checkpoint"].get("epoch"),
            "update": bundle["checkpoint"].get("optimizer_updates"),
            "scientific_config_sha256": scientific_config_sha256(config),
        },
        query_set={
            "split": "val",
            "event_count": int(core["query_count"]),
            "manifest_path": str(query_path.resolve()),
            "manifest_sha256": core["query_manifest_sha256"],
        },
        controls=controls,
    )
    summary.update(
        {
            "artifacts": {
                **artifacts,
                "donor_swap": donor_artifacts,
                "feature_cache": feature_cache,
            },
            "cached_donor_feature_equivalence": donor_feature_check,
            "clean_cached_vs_direct": clean_prediction_check,
            "clean_saved_evaluation_comparison": _saved_clean_comparison(
                config, encoded, query_ids, results["clean"]
            ),
            "descriptive_contrasts": _retrieval_differences(
                results["clean"],
                results["temporal_shift"],
                donor_aggregate,
                structure_metrics,
            ),
            "model_condition": bundle["model_condition"],
            "provenance": {
                "control_scope": "target_local_for_temporal_and_donor",
                "runtime_context": bundle["runtime_context"],
                "split": "val",
                "test_metrics_inspected": False,
                "test_model_evaluation": "not_run",
                "test_neural_data_opened": False,
                "test_predictions_generated": False,
            },
        }
    )
    summary_path = write_audit_summary(config, summary)
    detail_summary = save_audit_result(detail_dir / "summary.json", summary)
    return {
        "audit_summary": str(summary_path.resolve()),
        "audit_summary_sha256": file_sha256(summary_path),
        "detail_summary": str(detail_summary.resolve()),
        "detail_summary_sha256": file_sha256(detail_summary),
        "model_condition": bundle["model_condition"],
        "query_count": int(core["query_count"]),
        "selector": "/".join(
            (
                identity["dataset"],
                identity["subject_scope"],
                identity["experiment_id"],
            )
        ),
        "split": "val",
        "status": "completed",
        "test_model_evaluation": "not_run",
    }


def run_experiment(selector: str, *, device_name="auto", mapping_only=False) -> dict:
    """审计 exact selector；mapping-only 不读取 checkpoint 或执行模型 forward。"""
    record, config, adapter = _load_exact_config(selector)
    table = _validation_table(config, adapter)
    minimal = {
        "config": config,
        "event_path": Path(config["cache"]["event_table"]),
        "identity": record["identity"],
        "table": table,
    }
    mappings = _build_mappings(minimal)
    core = mappings["core_audit_queries.json"]
    result = {
        "checkpoint": str(
            (Path(config["training"]["output_dir"]) / "best.pt").resolve()
        ),
        "core_query_count": int(core["query_count"]),
        "dataset": record["identity"]["dataset"],
        "donor_eligible_count": int(core["donor_eligible_count"]),
        "event_table_sha256": file_sha256(minimal["event_path"]),
        "mapping_directory": str(
            _mapping_directory(record["identity"]["dataset"]).resolve()
        ),
        "model_condition": model_condition_from_config(config),
        "scientific_config_sha256": scientific_config_sha256(config),
        "selector": selector,
        "split": "val",
        "status": "mapping_only" if mapping_only else "validation_only",
        "temporal_eligible_count": int(core["temporal_eligible_count"]),
        "test_metrics_inspected": False,
        "test_model_evaluation": "not_run",
        "test_neural_data_opened": False,
        "test_predictions_generated": False,
    }
    if mapping_only:
        protocol = _load_protocol_assets(record["identity"]["dataset"])
        result["language"] = protocol["language"]
        result["vocabulary_status"] = {
            f"N{size}": (
                "frozen"
                if size in protocol["vocabularies"]
                else protocol["vocabulary_statuses"][size]["status"]
            )
            for size in PRIMARY_VOCABULARY_SIZES
        }
        return result

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    bundle = _build_loader_bundle(selector, device, load_model=True)
    protocol = _load_protocol_assets(record["identity"]["dataset"])
    return _run_full_audit(bundle, mappings, protocol, device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selector")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mapping-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_experiment(
        args.selector,
        device_name=args.device,
        mapping_only=args.mapping_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
