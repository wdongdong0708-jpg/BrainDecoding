"""在 validation 上运行冻结映射的 clean、temporal shift 与 donor swap。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from braindecoding.audit.controls import (
    DONOR_SEEDS,
    build_control_assets,
    file_sha256,
    write_control_assets,
)
from braindecoding.audit.evaluate import (
    collect_brain_features,
    evaluate_control,
    fixed_vocabulary_metrics_for_queries,
    predict_from_cached_features,
    save_audit_result,
    save_feature_cache,
    verify_cached_donor_features,
)
from braindecoding.config import PROJECT_ROOT
from braindecoding.data import chineseeeg2, smn4lang
from experiments.generate_manifests import validate_manifest_sha256
from models import build_brain_embedding_model
from tasks.word_decoding.ChineseEEG2_LittlePrince import evaluate as chinese_evaluate
from tasks.word_decoding.ChineseEEG2_LittlePrince import train as chinese_training
from tasks.word_decoding.SMN4Lang import train as smn_training


MANIFEST_ROOT = PROJECT_ROOT / "experiments" / "manifests"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "audits"
CHINESE_EVENT_TABLE = (
    PROJECT_ROOT
    / "tasks/word_decoding/ChineseEEG2_LittlePrince/cache"
    / "ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_semantic_v1.csv"
)
SMN_EVENT_TABLE = (
    PROJECT_ROOT
    / "tasks/word_decoding/SMN4Lang/cache"
    / "SMN4Lang_events_all_words_1s_fixed_3s_support.csv"
)
CONFIGS = {
    "ChineseEEG2": {
        "word": PROJECT_ROOT
        / "configs/ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_cnn_only.yaml",
        "neural_context": PROJECT_ROOT
        / "configs/ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_semantic_cnn_warm_start.yaml",
    },
    "SMN4Lang": {
        "word": PROJECT_ROOT / "configs/SMN4Lang_1s_conv_only.yaml",
        "neural_context": PROJECT_ROOT / "configs/SMN4Lang_1s.yaml",
    },
}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_protocol_assets(dataset_key: str) -> dict:
    directory = MANIFEST_ROOT / dataset_key
    vocabulary_manifests = {}
    for size in (20, 50, 100, 150):
        manifest = _load_json(directory / f"vocabulary_N{size}.json")
        validate_manifest_sha256(manifest)
        vocabulary_manifests[size] = manifest
    support = _load_json(directory / "ovmi_support.json")
    validate_manifest_sha256(support)
    return {
        "story_reference": _load_json(directory / "story_reference.json"),
        "support": support,
        "vocabularies": vocabulary_manifests,
    }


def _compare_event_contracts(reference, current, *, context_required: bool) -> None:
    """允许 word 忽略分组名称，但绝不允许脑窗口或目标事件漂移。"""
    columns = [
        "event_id",
        "subject_id",
        "recording_id",
        "normalized_word",
        "split",
        "window_start_seconds",
        "window_stop_seconds",
        "window_start_target_sample",
        "target_sample_count",
    ]
    if context_required:
        columns.append("sentence_uid")
    left = reference[columns].sort_values("event_id").reset_index(drop=True)
    right = current[columns].sort_values("event_id").reset_index(drop=True)
    if not left.equals(right):
        differences = [column for column in columns if not left[column].equals(right[column])]
        raise ValueError(f"审计事件表与 checkpoint 数据合同不同：{differences}")


def _load_chinese(model_condition: str, device):
    config = chinese_training.载入配置(CONFIGS["ChineseEEG2"][model_condition])
    event_path = chinese_training.确保事件表(config)
    table = chineseeeg2.载入事件表(event_path, split="val")
    reference = chineseeeg2.载入事件表(CHINESE_EVENT_TABLE, split="val")
    _compare_event_contracts(
        reference, table, context_required=model_condition == "neural_context"
    )
    dataset = chinese_training.构建数据集(config, table)
    loader, _ = chinese_training.make_loader(
        dataset, config["training"], shuffle=False
    )
    checkpoint_path = Path(config["training"]["output_dir"]) / "best.pt"
    checkpoint = chinese_training.load_checkpoint(checkpoint_path, map_location="cpu")
    vocabulary50 = _load_protocol_assets("chineseeeg2")["vocabularies"][50][
        "vocabulary"
    ]
    chinese_evaluate.验证检查点合同(checkpoint, config, dataset, vocabulary50)
    if bool(checkpoint["model_config"].get("use_transformer", True)) != (
        model_condition == "neural_context"
    ):
        raise ValueError("ChineseEEG2 checkpoint 的 word/context 模型条件不匹配。")
    model = build_brain_embedding_model(
        dataset.channel_count,
        dataset.channel_positions,
        dataset.subject_count,
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "dataset": dataset,
        "event_path": Path(event_path),
        "loader": loader,
        "model": model,
        "signal_key": "eeg",
        "subject_scope": "sub-01_to_sub-08",
        "table": table,
    }


def _validate_smn_checkpoint(checkpoint, config, dataset, model_condition):
    if checkpoint.get("task") != "word_decoding/SMN4Lang":
        raise ValueError("checkpoint 不属于 SMN4Lang。")
    if checkpoint.get("text_embedding_config") != config["text_embedding"]:
        raise ValueError("SMN4Lang checkpoint 文本向量合同不一致。")
    if checkpoint.get("model_config") != config["model"]:
        raise ValueError("SMN4Lang checkpoint 模型配置不一致。")
    expected_transformer = model_condition == "neural_context"
    if bool(checkpoint["model_config"].get("use_transformer", True)) != expected_transformer:
        raise ValueError("SMN4Lang checkpoint 的 word/context 模型条件不匹配。")
    if tuple(checkpoint.get("channel_names", ())) != tuple(dataset.channel_names):
        raise ValueError("SMN4Lang checkpoint 通道顺序不一致。")
    positions = np.asarray(checkpoint.get("channel_positions", ()))
    if positions.shape != dataset.channel_positions.shape or not np.allclose(
        positions, dataset.channel_positions, rtol=0.0, atol=1e-7
    ):
        raise ValueError("SMN4Lang checkpoint 传感器坐标不一致。")
    contract = checkpoint.get("dataset_contract", {})
    for key in (
        "split",
        "window_seconds",
        "target_sampling_rate_hz",
        "training_vocabulary_policy",
        "evaluation_vocabulary_policy",
        "context_grouping",
    ):
        if contract.get(key) != config["dataset"].get(key):
            raise ValueError(f"SMN4Lang checkpoint 数据合同不一致：{key}")


def _reuse_relocated_smn_validation_caches(dataset, config, table) -> dict:
    """仅在源文件搬家且内容签名不变时复用既有 validation MEG 缓存。"""
    root = Path(config["dataset"]["root"])
    cache_dir = Path(config["cache"]["meg_dir"])
    relocated = {}
    for relative_path in table["fif_relpath"].drop_duplicates():
        source_path = root / str(relative_path)
        current = smn4lang._preprocessing_signature(config["dataset"], source_path)
        current_without_path = {key: value for key, value in current.items() if key != "source_path"}
        matches = []
        for metadata_path in sorted(cache_dir.glob(f"{source_path.stem}_*.json")):
            metadata = _load_json(metadata_path)
            previous = metadata.get("signature", {})
            previous_without_path = {
                key: value for key, value in previous.items() if key != "source_path"
            }
            array_path = metadata_path.with_suffix(".npy")
            if previous_without_path == current_without_path and array_path.exists():
                matches.append(array_path)
        if len(matches) != 1:
            raise FileNotFoundError(
                f"无法为搬迁后的 validation 记录唯一匹配既有 MEG 缓存：{source_path}"
            )
        dataset.recording_cache_paths[str(relative_path)] = matches[0]
        relocated[str(relative_path)] = {
            "cache_path": str(matches[0].resolve()),
            "current_source_path": str(source_path.resolve()),
            "source_content_signature_equal": True,
        }
    dataset.zero_meg = False
    return relocated


def _load_smn(model_condition: str, device, dataset_root: Path | None):
    config = smn_training.load_config(CONFIGS["SMN4Lang"][model_condition])
    if dataset_root is not None:
        config["dataset"]["root"] = str(Path(dataset_root).resolve())
    event_path = smn_training.ensure_event_table(config)
    table = smn4lang.load_event_table(event_path, split="val")
    reference = smn4lang.load_event_table(SMN_EVENT_TABLE, split="val")
    _compare_event_contracts(
        reference, table, context_required=model_condition == "neural_context"
    )
    if dataset_root is None:
        dataset = smn_training.build_dataset(config, table)
        relocated_caches = {}
    else:
        # 本机数据目录曾移动；先按当前源文件核对 schema，再严格匹配旧缓存签名。
        dataset = smn_training.build_dataset(config, table, zero_meg=True)
        relocated_caches = _reuse_relocated_smn_validation_caches(dataset, config, table)
    loader, _ = smn_training.make_loader(dataset, config["training"], shuffle=False)
    checkpoint_path = Path(config["training"]["output_dir"]) / "best.pt"
    checkpoint = smn_training.load_checkpoint(checkpoint_path, map_location="cpu")
    _validate_smn_checkpoint(checkpoint, config, dataset, model_condition)
    model = build_brain_embedding_model(
        dataset.channel_count,
        dataset.channel_positions,
        dataset.subject_count,
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "dataset": dataset,
        "event_path": Path(event_path),
        "loader": loader,
        "model": model,
        "relocated_validation_caches": relocated_caches,
        "signal_key": "meg",
        "subject_scope": "development_sub01",
        "table": table,
    }


def _generic_metric_fields(metrics):
    fields = (
        "query_count",
        "candidate_count",
        "target_count",
        "median_rank",
        "mean_reciprocal_rank",
        "micro_recall_at_1",
        "micro_recall_at_10",
        "macro_recall_at_1",
        "macro_recall_at_10",
    )
    return {field: metrics[field] for field in fields}


def _compare_saved_clean(new_metrics, config) -> dict:
    path = Path(config["training"]["output_dir"]) / "evaluation_val.json"
    if not path.exists():
        return {"available": False, "reason": "saved_evaluation_val_missing"}
    old_metrics = _load_json(path)["metrics"]
    old = _generic_metric_fields(old_metrics)
    new = _generic_metric_fields(new_metrics)
    deltas = {key: float(new[key]) - float(old[key]) for key in old}
    exact = all(new[key] == old[key] for key in old)
    if not exact:
        raise AssertionError(f"新 clean runner 与旧 evaluate 不一致：{deltas}")
    return {
        "available": True,
        "exactly_equal": True,
        "field_deltas": deltas,
        "saved_evaluation": str(path.resolve()),
    }


def _run_model(
    dataset: str,
    dataset_key: str,
    model_condition: str,
    bundle: dict,
    mappings: dict[str, dict],
    protocol_assets: dict,
    output_root: Path,
    device,
) -> dict:
    model = bundle["model"]
    config = bundle["config"]
    checkpoint_path = bundle["checkpoint_path"]
    checkpoint_sha = file_sha256(checkpoint_path)
    core = mappings["core_audit_queries.json"]
    query_event_ids = core["event_ids"]
    encoded = collect_brain_features(
        model,
        bundle["loader"],
        device,
        signal_key=bundle["signal_key"],
        amp=config["training"].get("amp", True),
    )
    feature_dir = output_root / model_condition / "feature_cache"
    feature_cache = save_feature_cache(
        feature_dir,
        encoded,
        {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "event_table_sha256": file_sha256(bundle["event_path"]),
            "model_condition": model_condition,
            "split": "val",
            "subject_scope": bundle["subject_scope"],
            "test_neural_data_opened": False,
        },
    )
    feature_equivalence = verify_cached_donor_features(
        model,
        bundle["dataset"],
        encoded,
        mappings["donor_swap_seed00.json"],
        device,
        signal_key=bundle["signal_key"],
        amp=config["training"].get("amp", True),
    )

    controls = [
        ("clean", mappings["clean_identity_mapping.json"]),
        ("temporal_shift", mappings["temporal_shift_mapping.json"]),
    ]
    controls.extend(
        (f"donor_swap_seed{seed:02d}", mappings[f"donor_swap_seed{seed:02d}.json"])
        for seed in DONOR_SEEDS
    )
    result_paths = {}
    clean_metrics = None
    for control_name, mapping in controls:
        predictions, mapping_audit = predict_from_cached_features(
            model,
            encoded,
            mapping,
            query_event_ids,
            model_condition=model_condition,
            device=device,
            amp=config["training"].get("amp", True),
        )
        metrics = evaluate_control(
            predictions,
            encoded,
            query_event_ids,
            protocol_assets["vocabularies"],
            protocol_assets["support"],
            protocol_assets["story_reference"],
            dataset_name=dataset,
        )
        if control_name == "clean":
            clean_metrics = metrics["N50"]
        seed = mapping.get("seed")
        result = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "control_condition": control_name,
            "control_event_table_sha256": mapping["event_table_sha256"],
            "control_mapping_sha256": mapping["mapping_sha256"],
            "dataset": dataset,
            "development_only": dataset == "SMN4Lang",
            "event_table": str(bundle["event_path"].resolve()),
            "event_table_sha256": file_sha256(bundle["event_path"]),
            "feature_cache": feature_cache,
            "mapping_audit": mapping_audit,
            "metrics": metrics,
            "model_condition": model_condition,
            "query_count": int(core["query_count"]),
            "query_manifest_sha256": core["query_manifest_sha256"],
            "seed": seed,
            "seeds": list(DONOR_SEEDS) if seed is None else None,
            "split": "val",
            "status": "development_only" if dataset == "SMN4Lang" else "validation_only",
            "subject_scope": bundle["subject_scope"],
            "test_eeg_opened": False,
            "test_meg_opened": False,
            "vocabulary_manifest_sha256": {
                f"N{size}": manifest["manifest_sha256"]
                for size, manifest in protocol_assets["vocabularies"].items()
            },
        }
        path = save_audit_result(
            output_root / model_condition / f"{control_name}.json", result
        )
        result_paths[control_name] = str(path.resolve())

    clean_equivalence = _compare_saved_clean(clean_metrics, config)
    summary = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_contract": "validated",
        "checkpoint_sha256": checkpoint_sha,
        "clean_existing_evaluate_equivalence": clean_equivalence,
        "control_results": result_paths,
        "dataset": dataset,
        "development_only": dataset == "SMN4Lang",
        "feature_cache": feature_cache,
        "feature_direct_forward_equivalence": feature_equivalence,
        "relocated_validation_caches": bundle.get("relocated_validation_caches", {}),
        "model_condition": model_condition,
        "query_count": int(core["query_count"]),
        "query_manifest_sha256": core["query_manifest_sha256"],
        "split": "val",
        "subject_scope": bundle["subject_scope"],
        "test_eeg_opened": False,
        "test_meg_opened": False,
    }
    save_audit_result(output_root / model_condition / "summary.json", summary)
    return summary


def run_dataset(
    dataset: str,
    *,
    model_conditions=("word", "neural_context"),
    device_name="auto",
    smn_dataset_root=None,
    mapping_only=False,
) -> dict:
    """只运行一个数据集的 validation 控制，硬编码拒绝 test。"""
    if dataset not in {"ChineseEEG2", "SMN4Lang"}:
        raise ValueError(f"未知数据集：{dataset}")
    dataset_key = dataset.lower()
    event_path = CHINESE_EVENT_TABLE if dataset == "ChineseEEG2" else SMN_EVENT_TABLE
    event_sha = file_sha256(event_path)
    table = pd.read_csv(event_path, low_memory=False)
    mappings = build_control_assets(
        table,
        dataset=dataset,
        event_table_sha256=event_sha,
        split="val",
        status="validation_only" if dataset == "ChineseEEG2" else "development_only",
    )
    output_root = OUTPUT_ROOT / dataset_key / "validation"
    write_control_assets(mappings, output_root / "mappings")
    core = mappings["core_audit_queries.json"]
    result = {
        "core_query_count": core["query_count"],
        "dataset": dataset,
        "development_only": dataset == "SMN4Lang",
        "donor_eligible_count": core["donor_eligible_count"],
        "event_table_sha256": event_sha,
        "mapping_directory": str((output_root / "mappings").resolve()),
        "models": {},
        "split": "val",
        "status": "validation_only" if dataset == "ChineseEEG2" else "development_only",
        "temporal_eligible_count": core["temporal_eligible_count"],
        "test_neural_data_opened": False,
    }
    if mapping_only:
        return result

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    protocol_assets = _load_protocol_assets(dataset_key)
    for model_condition in model_conditions:
        if model_condition not in {"word", "neural_context"}:
            raise ValueError(f"未知模型条件：{model_condition}")
        if dataset == "ChineseEEG2":
            bundle = _load_chinese(model_condition, device)
        else:
            bundle = _load_smn(model_condition, device, smn_dataset_root)
        result["models"][model_condition] = _run_model(
            dataset,
            dataset_key,
            model_condition,
            bundle,
            mappings,
            protocol_assets,
            output_root,
            device,
        )
    save_audit_result(output_root / "run_summary.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("chineseeeg2", "smn4lang", "all"), default="all"
    )
    parser.add_argument(
        "--model", choices=("word", "neural_context", "all"), default="all"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smn-dataset-root", type=Path)
    parser.add_argument("--mapping-only", action="store_true")
    args = parser.parse_args(argv)
    datasets = (
        ("ChineseEEG2", "SMN4Lang")
        if args.dataset == "all"
        else (("ChineseEEG2",) if args.dataset == "chineseeeg2" else ("SMN4Lang",))
    )
    model_conditions = (
        ("word", "neural_context") if args.model == "all" else (args.model,)
    )
    for dataset in datasets:
        result = run_dataset(
            dataset,
            model_conditions=model_conditions,
            device_name=args.device,
            smn_dataset_root=args.smn_dataset_root,
            mapping_only=args.mapping_only,
        )
        print(
            f"{dataset}: temporal={result['temporal_eligible_count']}, "
            f"donor={result['donor_eligible_count']}, "
            f"core={result['core_query_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
