"""从统一 brain feature 缓存评价 validation neural controls。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from braindecoding.audit.controls import (
    file_sha256,
    json_bytes,
    validate_payload_sha256,
    with_payload_sha256,
)
from braindecoding.evaluation.ovmi import full_ovmi_metrics
from braindecoding.evaluation.retrieval import (
    normalize_rows,
    retrieval_ranks,
    summarize_retrieval,
    unique_candidates,
)


@torch.inference_mode()
def collect_brain_features(
    model,
    loader,
    device,
    *,
    signal_key: str,
    amp=True,
    collect_direct_predictions=False,
) -> dict:
    """缓存 brain feature，并可在同一次 encoder forward 中保留直接预测。"""
    model.eval()
    features = []
    targets = []
    subject_indices = []
    sentence_indices = []
    words = []
    event_ids = []
    recording_ids = []
    direct_predictions = []
    batch_slices = []
    use_amp = bool(amp) and device.type == "cuda"
    start = 0
    for batch in loader:
        signals = batch[signal_key].to(device, non_blocking=True)
        subjects = batch["subject_index"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            if collect_direct_predictions:
                direct, feature = model(
                    signals,
                    subjects,
                    batch["sentence_index"].to(device, non_blocking=True),
                    return_brain_embedding=True,
                )
                direct_predictions.append(direct.float().cpu())
            else:
                feature = F.normalize(model.brain_encoder(signals, subjects), dim=-1)
        features.append(feature.cpu())
        targets.append(batch["text_embedding"].float().cpu())
        subject_indices.append(batch["subject_index"].long().cpu())
        sentence_indices.append(batch["sentence_index"].long().cpu())
        words.extend(str(word) for word in batch["word"])
        event_ids.extend(str(value) for value in batch["event_id"])
        recording_ids.extend(str(value) for value in batch["recording_id"])
        stop = start + len(feature)
        batch_slices.append((start, stop))
        start = stop
    if not features:
        raise ValueError("validation loader 为空。")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("feature cache 中出现重复事件编号。")
    result = {
        "batch_slices": batch_slices,
        "brain_features": torch.cat(features),
        "event_ids": event_ids,
        "recording_ids": recording_ids,
        "sentence_indices": torch.cat(sentence_indices),
        "subject_indices": torch.cat(subject_indices),
        "target_embeddings": torch.cat(targets),
        "words": words,
    }
    if collect_direct_predictions:
        result["direct_predictions"] = torch.cat(direct_predictions)
    return result


def _source_by_target(mapping: dict) -> dict[str, str]:
    validate_payload_sha256(mapping, "mapping_sha256")
    if mapping.get("split") != "val":
        raise ValueError("控制映射必须是 validation-only。")
    result = {}
    for row in mapping["mappings"]:
        target = str(row["target_event_id"])
        if target in result:
            raise ValueError(f"控制映射重复定义 target：{target}")
        result[target] = str(row["source_event_id"])
    return result


def apply_feature_mapping(
    encoded: dict,
    mapping: dict,
    query_event_ids,
) -> tuple[torch.Tensor, dict]:
    """只替换共同查询位置的 feature，其他上下文位置保持 clean。"""
    event_index = {
        event_id: index for index, event_id in enumerate(encoded["event_ids"])
    }
    source_by_target = _source_by_target(mapping)
    query_event_ids = [str(value) for value in query_event_ids]
    missing_queries = sorted(set(query_event_ids) - set(event_index))
    if missing_queries:
        raise ValueError(f"feature cache 缺少 core query：{missing_queries[:5]}")
    missing_mappings = sorted(set(query_event_ids) - set(source_by_target))
    if missing_mappings:
        raise ValueError(f"控制映射缺少 core query：{missing_mappings[:5]}")
    missing_sources = sorted(set(source_by_target.values()) - set(event_index))
    if missing_sources:
        raise ValueError(f"feature cache 缺少 source event：{missing_sources[:5]}")

    controlled = encoded["brain_features"].clone()
    original = encoded["brain_features"]
    for target_id in query_event_ids:
        target_index = event_index[target_id]
        source_index = event_index[source_by_target[target_id]]
        controlled[target_index] = original[source_index]
    return controlled, {
        "context_group_indices_unchanged": True,
        "non_core_positions_left_clean": True,
        "query_count": len(query_event_ids),
        "target_positions_unchanged": True,
    }


@torch.inference_mode()
def predict_from_cached_features(
    model,
    encoded: dict,
    mapping: dict,
    query_event_ids,
    *,
    model_condition: str,
    device,
    amp=True,
) -> tuple[torch.Tensor, dict]:
    """在原批次/上下文位置上运行 word 或 grouped neural_context。"""
    if model_condition not in {"word", "neural_context"}:
        raise ValueError(f"未知模型条件：{model_condition}")
    if model_condition == "word" and model.context_transformer is not None:
        raise ValueError("word 条件必须使用 model.use_transformer=false。")
    if model_condition == "neural_context":
        if model.context_transformer is None or model.context_mode != "grouped":
            raise ValueError("neural_context 必须使用 grouped context transformer。")

    model.eval()
    controlled, mapping_audit = apply_feature_mapping(
        encoded, mapping, query_event_ids
    )
    predictions = []
    use_amp = bool(amp) and device.type == "cuda"
    for start, stop in encoded["batch_slices"]:
        batch_features = controlled[start:stop].to(device, non_blocking=True)
        group_indices = encoded["sentence_indices"][start:stop].to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            if model_condition == "word":
                output = batch_features
            else:
                output = model.context_transformer(batch_features, group_indices)
                output = F.normalize(output, dim=-1)
        predictions.append(output.float().cpu())
    return torch.cat(predictions), mapping_audit


@torch.inference_mode()
def predict_structure_only_from_cached_features(
    model,
    encoded: dict,
    *,
    model_condition: str,
    device,
    amp=True,
) -> tuple[torch.Tensor | None, dict]:
    """把整个 batch 的神经 feature 清零，仅保留冻结结构与位置计算。"""
    if model_condition == "word":
        return None, {
            "status": "not_applicable",
            "reason": "no_context_transformer",
        }
    if model_condition != "neural_context":
        raise ValueError(f"未知模型条件：{model_condition}")
    if model.context_transformer is None or model.context_mode != "grouped":
        raise ValueError("structure_only 只支持 grouped context transformer。")

    model.eval()
    zero_features = torch.zeros_like(encoded["brain_features"])
    predictions = []
    use_amp = bool(amp) and device.type == "cuda"
    for start, stop in encoded["batch_slices"]:
        batch_features = zero_features[start:stop].to(device, non_blocking=True)
        group_indices = encoded["sentence_indices"][start:stop].to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            output = model.context_transformer(batch_features, group_indices)
            output = F.normalize(output, dim=-1)
        predictions.append(output.float().cpu())
    return torch.cat(predictions), {
        "status": "completed",
        "definition": "no_neural_feature_structural_inference_control",
        "all_context_brain_features_zero": bool(
            torch.count_nonzero(zero_features).item() == 0
        ),
        "batch_slices_unchanged": True,
        "context_group_indices_unchanged": True,
        "group_length_and_order_unchanged": True,
    }


def verify_cached_clean_predictions(
    model,
    encoded: dict,
    clean_mapping: dict,
    query_event_ids,
    *,
    model_condition: str,
    device,
    amp=True,
    atol=2e-5,
    rtol=2e-4,
) -> dict:
    """验证 cached-feature clean 与同批次 canonical direct forward 等价。"""
    if "direct_predictions" not in encoded:
        raise ValueError("feature cache 未保留 canonical direct predictions。")
    cached, _ = predict_from_cached_features(
        model,
        encoded,
        clean_mapping,
        query_event_ids,
        model_condition=model_condition,
        device=device,
        amp=amp,
    )
    direct = encoded["direct_predictions"].float()
    differences = (cached.float() - direct).abs()
    equivalent = torch.allclose(cached.float(), direct, atol=atol, rtol=rtol)
    if not equivalent:
        raise AssertionError(
            "cached-feature clean 与 canonical direct forward 不等价："
            f"max_abs={float(differences.max())}"
        )
    return {
        "allclose": True,
        "atol": float(atol),
        "event_ids_equal": True,
        "maximum_absolute_difference": float(differences.max().cpu()),
        "mean_absolute_difference": float(differences.mean().cpu()),
        "rtol": float(rtol),
        "shape": list(cached.shape),
    }


def fixed_vocabulary_metrics_for_queries(
    predictions,
    encoded: dict,
    query_event_ids,
    vocabulary,
    *,
    vocabulary_name: str,
    top_ks=(1, 10),
) -> tuple[dict, dict]:
    """候选仍由完整 validation 产生，只在共同查询集汇总指标。"""
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(encoded["target_embeddings"], dtype=np.float32)
    words = np.asarray([str(word).lower() for word in encoded["words"]])
    vocabulary_order = [str(word).lower() for word in vocabulary]
    vocabulary_set = set(vocabulary_order)
    all_selected = np.flatnonzero(np.isin(words, vocabulary_order))
    if len(all_selected) == 0:
        raise ValueError("validation 中没有冻结词表查询。")
    candidates, candidate_words = unique_candidates(
        targets[all_selected], words[all_selected]
    )

    event_index = {
        event_id: index for index, event_id in enumerate(encoded["event_ids"])
    }
    query_indices = np.asarray(
        [
            event_index[str(event_id)]
            for event_id in query_event_ids
            if words[event_index[str(event_id)]] in vocabulary_set
        ],
        dtype=np.int64,
    )
    if len(query_indices) == 0:
        raise ValueError("core audit query 中没有冻结词表查询。")
    true_words = words[query_indices]
    ranks = retrieval_ranks(
        predictions[query_indices], true_words, candidates, candidate_words
    )
    metrics = summarize_retrieval(
        ranks, true_words, len(candidate_words), top_ks=top_ks
    )
    metrics["vocabulary_size"] = len(vocabulary_order)
    metrics["observed_vocabulary_size"] = len(candidate_words)
    metrics["missing_vocabulary_words"] = sorted(
        vocabulary_set - set(candidate_words)
    )
    for k in top_ks:
        metrics[f"retrieval_acc{k}_vocab={vocabulary_name}"] = metrics[
            f"micro_recall_at_{k}"
        ]
        metrics[f"retrieval_acc{k}_vocab={vocabulary_name}_macro"] = metrics[
            f"macro_recall_at_{k}"
        ]

    normalized_queries = normalize_rows(predictions[query_indices])
    normalized_candidates = normalize_rows(candidates)
    predicted_words = np.asarray(candidate_words)[
        np.argmax(normalized_queries @ normalized_candidates.T, axis=1)
    ]
    details = {
        "predicted_words": [str(word) for word in predicted_words],
        "query_indices": query_indices,
        "ranks": ranks,
        "true_words": [str(word) for word in true_words],
    }
    return metrics, details


def evaluate_control(
    predictions,
    encoded: dict,
    query_event_ids,
    vocabulary_manifests: dict[int, dict],
    ovmi_support_manifest: dict | None,
    story_reference: dict | None,
    *,
    dataset_name: str,
    language: str,
    vocabulary_statuses: dict[int, dict] | None = None,
) -> dict:
    """按实际冻结词表汇总 retrieval；缺失资产保持显式 not_run。"""
    results = {}
    for size, vocabulary_manifest in sorted(vocabulary_manifests.items()):
        vocabulary = vocabulary_manifest["vocabulary"]
        metrics, details = fixed_vocabulary_metrics_for_queries(
            predictions,
            encoded,
            query_event_ids,
            vocabulary,
            vocabulary_name=f"{dataset_name.lower()}_N{size}",
        )
        frozen_support = None
        if ovmi_support_manifest is not None:
            frozen_support = (
                ovmi_support_manifest.get("vocabularies", {})
                .get(f"N{size}", {})
                .get("splits", {})
                .get("val")
            )
        missing_words = list(metrics["missing_vocabulary_words"])
        if story_reference is None:
            metrics["ovmi_story"] = {
                "available": False,
                "reason": "story_reference_not_frozen",
                "status": "not_run",
            }
        elif not missing_words:
            metrics["ovmi_story"] = full_ovmi_metrics(
                details["true_words"],
                details["predicted_words"],
                vocabulary,
                {
                    "enabled": True,
                    "method": "full",
                    "reference": story_reference,
                    "language": str(language),
                },
            )
        else:
            metrics["ovmi_story"] = {
                "available": False,
                "frozen_validation_support_status": (
                    frozen_support.get("full_ovmi_status")
                    if frozen_support is not None
                    else None
                ),
                "missing_word_count": len(missing_words),
                "missing_words": missing_words,
                "reason": "missing_true_class_support",
                "supported_word_count": len(vocabulary) - len(missing_words),
            }
        metrics["ovmi_domain"] = {
            "available": False,
            "reason": "domain_reference_not_frozen",
        }
        results[f"N{size}"] = metrics
    for size, status in sorted((vocabulary_statuses or {}).items()):
        results.setdefault(f"N{int(size)}", dict(status))
    return results


_AGGREGATE_RETRIEVAL_FIELDS = {
    "top1": "micro_recall_at_1",
    "top10": "micro_recall_at_10",
    "macro_top1": "macro_recall_at_1",
    "macro_top10": "macro_recall_at_10",
    "median_rank": "median_rank",
    "mrr": "mean_reciprocal_rank",
}


def _summary_statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(len(values))
    return {
        "count": int(len(values)),
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half_width),
        "ci95_high": float(mean + half_width),
    }


def aggregate_donor_results(per_seed_results: dict[str, dict]) -> dict:
    """按固定 20 seeds 汇总 donor retrieval 与可用 story OVMI。"""
    if len(per_seed_results) != 20:
        raise ValueError("donor aggregate 必须包含固定的 20 个 seed。")
    sizes = sorted(
        set.intersection(
            *(set(result) for result in per_seed_results.values())
        )
    )
    aggregate = {}
    for size in sizes:
        blocks = [result[size] for result in per_seed_results.values()]
        if any(block.get("status") == "not_run" for block in blocks):
            aggregate[size] = {
                "status": "not_run",
                "reason": blocks[0].get(
                    "reason", "vocabulary_manifest_not_frozen"
                ),
            }
            continue
        retrieval = {
            name: _summary_statistics([block[field] for block in blocks])
            for name, field in _AGGREGATE_RETRIEVAL_FIELDS.items()
        }
        ovmi = [block.get("ovmi_story", {}) for block in blocks]
        if all(item.get("available") and item.get("score_bits") is not None for item in ovmi):
            ovmi_story = {
                "status": "completed",
                "score_bits": _summary_statistics(
                    [item["score_bits"] for item in ovmi]
                ),
            }
        else:
            ovmi_story = {
                "status": "unavailable",
                "reason": "not_available_for_all_donor_seeds",
            }
        aggregate[size] = {
            "status": "completed",
            "retrieval": retrieval,
            "ovmi_story": ovmi_story,
        }
    return aggregate


def save_feature_cache(output_dir, encoded: dict, provenance: dict) -> dict:
    """用稳定 NPY 与 JSON 保存一次性 brain encoder 输出。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "brain_features.npy"
    target_path = output_dir / "target_embeddings.npy"
    np.save(feature_path, encoded["brain_features"].cpu().numpy(), allow_pickle=False)
    np.save(target_path, encoded["target_embeddings"].cpu().numpy(), allow_pickle=False)
    index = with_payload_sha256(
        {
            **dict(provenance),
            "batch_slices": [list(value) for value in encoded["batch_slices"]],
            "brain_features_sha256": file_sha256(feature_path),
            "event_ids": list(encoded["event_ids"]),
            "recording_ids": list(encoded["recording_ids"]),
            "sentence_indices": encoded["sentence_indices"].tolist(),
            "subject_indices": encoded["subject_indices"].tolist(),
            "target_embeddings_sha256": file_sha256(target_path),
            "words": list(encoded["words"]),
        },
        "feature_index_sha256",
    )
    index_path = output_dir / "feature_index.json"
    index_path.write_bytes(json_bytes(index))
    return {
        "brain_features": str(feature_path.resolve()),
        "brain_features_sha256": index["brain_features_sha256"],
        "feature_index": str(index_path.resolve()),
        "feature_index_sha256": index["feature_index_sha256"],
        "target_embeddings": str(target_path.resolve()),
        "target_embeddings_sha256": index["target_embeddings_sha256"],
    }


@torch.inference_mode()
def verify_cached_donor_features(
    model,
    dataset,
    encoded: dict,
    mapping: dict,
    device,
    *,
    signal_key: str,
    maximum_sources=16,
    amp=True,
    atol=2e-5,
    rtol=2e-4,
) -> dict:
    """重新读取 donor 窗口并前向，核对缓存 feature 的数值等价性。"""
    validate_payload_sha256(mapping, "mapping_sha256")
    source_ids = list(
        dict.fromkeys(row["source_event_id"] for row in mapping["mappings"])
    )[: int(maximum_sources)]
    dataset_index = {
        str(event_id): index
        for index, event_id in enumerate(dataset.table["event_id"].astype(str))
    }
    encoded_index = {
        event_id: index for index, event_id in enumerate(encoded["event_ids"])
    }
    use_amp = bool(amp) and device.type == "cuda"
    model.eval()
    selected_indices = {encoded_index[event_id] for event_id in source_ids}
    direct_by_index = {}
    # CUDA 卷积对 batch 形状可能产生微小舍入差；按原批次原顺序重放，
    # 才是在相同评价条件下核对“重读窗口”和缓存 feature。
    for start, stop in encoded["batch_slices"]:
        wanted = sorted(selected_indices.intersection(range(start, stop)))
        if not wanted:
            continue
        batch_event_ids = encoded["event_ids"][start:stop]
        items = [dataset[dataset_index[event_id]] for event_id in batch_event_ids]
        signals = torch.stack([item[signal_key] for item in items]).to(device)
        subjects = torch.stack([item["subject_index"] for item in items]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            direct_batch = F.normalize(model.brain_encoder(signals, subjects), dim=-1)
        for index in wanted:
            direct_by_index[index] = direct_batch[index - start]
    direct = torch.stack([direct_by_index[encoded_index[event_id]] for event_id in source_ids])
    cached = encoded["brain_features"][[encoded_index[event_id] for event_id in source_ids]].to(device)
    differences = (direct.float() - cached.float()).abs()
    equivalent = torch.allclose(direct.float(), cached.float(), atol=atol, rtol=rtol)
    if not equivalent:
        raise AssertionError(
            "缓存 donor feature 与重新读取窗口的 brain encoder 输出不等价："
            f"max_abs={float(differences.max())}"
        )
    return {
        "atol": float(atol),
        "equivalent": True,
        "maximum_absolute_difference": float(differences.max().cpu()),
        "rtol": float(rtol),
        "source_count": len(source_ids),
    }


def save_audit_result(path, result: dict) -> Path:
    """保存不带时间戳的稳定 audit 结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = with_payload_sha256(result, "result_sha256")
    path.write_bytes(json_bytes(payload))
    return path
