"""LibriBrain100 Sherlock1 词事件适配器与缓存工具。

事件表构建器只读取 TSV 标注和 H5 元数据。MEG 信号物化与 T5 向量物化是
两个显式且相互独立的阶段。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LIBRIBRAIN100_50_WORD_VOCABULARY = (
    "is",
    "the",
    "a",
    "to",
    "it",
    "i",
    "not",
    "was",
    "we",
    "be",
    "he",
    "that",
    "have",
    "this",
    "they",
    "of",
    "there",
    "and",
    "are",
    "in",
    "but",
    "will",
    "so",
    "all",
    "my",
    "for",
    "she",
    "were",
    "any",
    "really",
    "at",
    "out",
    "our",
    "am",
    "its",
    "had",
    "him",
    "an",
    "very",
    "has",
    "do",
    "can",
    "time",
    "think",
    "good",
    "always",
    "new",
    "people",
    "as",
    "on",
)

RECORDING_PATTERN = re.compile(
    r"sub-(?P<subject>[^_]+)_ses-(?P<session>\d+)_task-(?P<task>[^_]+)_"
    r"run-(?P<run>\d+)"
)


def normalize_word(value: str) -> str:
    """按照源实验规则，将单词清洗为小写字母数字形式。"""
    return "".join(
        character
        for character in str(value).strip()
        if character.isalnum() or character in {"-", "'"}
    ).lower()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decode_h5_attribute(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(value).tolist()
    )


def _configured_split_by_session(split_config) -> dict[int, str]:
    mapping = {}
    for split_name in ("train", "val", "test"):
        key = f"{split_name}_sessions"
        sessions = [int(value) for value in split_config.get(key, [])]
        for session in sessions:
            if session in mapping:
                raise ValueError(f"session {session} 被重复分配到多个数据划分。")
            mapping[session] = split_name
    if not mapping:
        raise ValueError("split 配置没有提供 train/val/test sessions。")
    return mapping


def _recording_files(dataset_config) -> list[dict]:
    root = Path(dataset_config["root"])
    task = str(dataset_config.get("task", "Sherlock1"))
    task_root = root / task
    events_root = task_root / dataset_config.get(
        "events_dir", "derivatives/events"
    )
    h5_root = task_root / dataset_config.get(
        "serialised_dir", "derivatives/serialised"
    )
    if not events_root.exists() or not h5_root.exists():
        raise FileNotFoundError(
            f"LibriBrain100 目录不完整：需要 {events_root} 和 {h5_root}。"
        )

    configured_sessions = sorted(
        _configured_split_by_session(dataset_config["split"])
    )
    records = []
    for session in configured_sessions:
        event_matches = sorted(events_root.glob(f"sub-*_ses-{session}_task-{task}_run-*_events.tsv"))
        h5_matches = sorted(h5_root.glob(f"sub-*_ses-{session}_task-{task}_run-*_meg.h5"))
        if len(event_matches) != 1 or len(h5_matches) != 1:
            raise FileNotFoundError(
                f"session {session} 应各有一份 events.tsv 和 H5；"
                f"实际为 {len(event_matches)} / {len(h5_matches)}。"
            )
        event_match = RECORDING_PATTERN.search(event_matches[0].name)
        h5_match = RECORDING_PATTERN.search(h5_matches[0].name)
        if event_match is None or h5_match is None:
            raise ValueError(f"无法解析 session {session} 的文件名。")
        event_key = tuple(event_match.group(key) for key in ("subject", "session", "task", "run"))
        h5_key = tuple(h5_match.group(key) for key in ("subject", "session", "task", "run"))
        if event_key != h5_key:
            raise ValueError(f"session {session} 的事件表与 H5 recording key 不一致。")
        records.append(
            {
                "subject": event_match.group("subject"),
                "session": session,
                "task": task,
                "run": int(event_match.group("run")),
                "events_path": event_matches[0],
                "h5_path": h5_matches[0],
                "task_root": task_root,
            }
        )
    return records


def inspect_h5_schema(path) -> dict:
    """只读取 H5 的形状和属性，不加载 MEG 信号数组。"""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"H5 缺少 data 数据集：{path}")
        data = handle["data"]
        if data.ndim != 2:
            raise ValueError(f"H5 data 应为 通道×时间，实际 {data.shape}：{path}")
        sampling_rate = float(handle.attrs["sample_frequency"])
        channel_names = _decode_h5_attribute(handle.attrs.get("channel_names"))
        channel_types = _decode_h5_attribute(handle.attrs.get("channel_types"))
        schema = {
            "channel_count": int(data.shape[0]),
            "sample_count": int(data.shape[1]),
            "sampling_rate_hz": sampling_rate,
            "channel_names": channel_names,
            "channel_types": channel_types,
        }
    if channel_names and len(channel_names) != schema["channel_count"]:
        raise ValueError(f"channel_names 数量与 data 不一致：{path}")
    if channel_types and len(channel_types) != schema["channel_count"]:
        raise ValueError(f"channel_types 数量与 data 不一致：{path}")
    return schema


def build_event_table(config) -> pd.DataFrame:
    """按照冻结的 10/1/1 划分，为每个标注词构建一行记录。"""
    split_by_session = _configured_split_by_session(config["split"])
    window_seconds = float(config.get("window_seconds", 3.0))
    eligibility_window_seconds = float(
        config.get("eligibility_window_seconds", window_seconds)
    )
    if eligibility_window_seconds < window_seconds:
        raise ValueError(
            "eligibility_window_seconds cannot be shorter than window_seconds."
        )
    target_rate = float(config.get("target_sampling_rate_hz", 50.0))
    expected_source_rate = float(config.get("source_sampling_rate_hz", 250.0))
    vocabulary = set(LIBRIBRAIN100_50_WORD_VOCABULARY)
    tables = []

    for record in _recording_files(config):
        schema = inspect_h5_schema(record["h5_path"])
        source_rate = schema["sampling_rate_hz"]
        if not math.isclose(source_rate, expected_source_rate):
            raise ValueError(
                f"采样率不符合配置：{record['h5_path']} 为 {source_rate} Hz，"
                f"预期 {expected_source_rate} Hz。"
            )
        recording_duration = schema["sample_count"] / source_rate
        raw = pd.read_csv(record["events_path"], sep="\t")
        required = {"kind", "segment", "timemeg", "duration", "sentenceidx", "wordidx"}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"事件表缺少字段 {missing}：{record['events_path']}")

        words = raw.loc[raw["kind"].eq("word")].copy()
        words = words.dropna(subset=["segment", "timemeg"])
        words["word"] = words["segment"].astype(str).str.strip()
        words["normalized_word"] = words["word"].map(normalize_word)
        words["onset_seconds"] = pd.to_numeric(words["timemeg"], errors="coerce")
        words["word_duration_seconds"] = pd.to_numeric(
            words["duration"], errors="coerce"
        ).fillna(0.0)
        words["sentence_index"] = pd.to_numeric(
            words["sentenceidx"], errors="coerce"
        ).fillna(0).astype(int)
        words["word_index"] = pd.to_numeric(
            words["wordidx"], errors="coerce"
        ).fillna(0).astype(int)

        recording_id = (
            f"sub-{record['subject']}_ses-{record['session']}_"
            f"task-{record['task']}_run-{record['run']}"
        )
        words["dataset"] = "LibriBrain100"
        words["subject_id"] = f"sub-{record['subject']}"
        words["task"] = record["task"]
        words["session"] = record["session"]
        words["run"] = record["run"]
        words["recording_id"] = recording_id
        words["split"] = split_by_session[record["session"]]
        words["sentence_uid"] = words["sentence_index"].map(
            lambda value: f"{recording_id}_sentence-{value}"
        )
        words["word_id"] = words["normalized_word"].map(_sha256_text)
        words["event_id"] = [
            _sha256_text(f"{recording_id}|{index}|{word}|{onset:.6f}")
            for index, word, onset in zip(
                words.index, words["normalized_word"], words["onset_seconds"]
            )
        ]
        words["source_sampling_rate_hz"] = source_rate
        words["source_sample_count"] = schema["sample_count"]
        words["recording_duration_seconds"] = recording_duration
        words["window_start_seconds"] = words["onset_seconds"]
        words["window_stop_seconds"] = words["onset_seconds"] + window_seconds
        words["eligibility_window_stop_seconds"] = (
            words["onset_seconds"] + eligibility_window_seconds
        )
        words["window_start_source_sample"] = np.rint(
            words["onset_seconds"] * source_rate
        ).astype(int)
        words["window_stop_source_sample"] = (
            words["window_start_source_sample"] + round(window_seconds * source_rate)
        )
        words["window_start_target_sample"] = np.rint(
            words["onset_seconds"] * target_rate
        ).astype(int)
        words["window_stop_target_sample"] = (
            words["window_start_target_sample"] + round(window_seconds * target_rate)
        )
        words["target_sampling_rate_hz"] = target_rate
        words["target_sample_count"] = round(window_seconds * target_rate)
        words["h5_relpath"] = record["h5_path"].relative_to(
            Path(config["root"])
        ).as_posix()
        words["events_relpath"] = record["events_path"].relative_to(
            Path(config["root"])
        ).as_posix()
        words["in_libribrain50"] = words["normalized_word"].isin(vocabulary)
        words["window_complete"] = (
            words["eligibility_window_stop_seconds"] <= recording_duration + 1e-9
        )
        words["is_trainable"] = (
            words["normalized_word"].ne("")
            & words["onset_seconds"].notna()
            & words["window_complete"]
        )
        words["exclusion_reason"] = np.where(
            words["normalized_word"].eq(""),
            "empty_normalized_word",
            np.where(~words["window_complete"], "incomplete_window", ""),
        )
        tables.append(
            words[
                [
                    "dataset",
                    "subject_id",
                    "task",
                    "session",
                    "run",
                    "recording_id",
                    "split",
                    "sentence_index",
                    "sentence_uid",
                    "word_index",
                    "word",
                    "normalized_word",
                    "word_id",
                    "event_id",
                    "onset_seconds",
                    "word_duration_seconds",
                    "source_sampling_rate_hz",
                    "source_sample_count",
                    "recording_duration_seconds",
                    "window_start_seconds",
                    "window_stop_seconds",
                    "eligibility_window_stop_seconds",
                    "window_start_source_sample",
                    "window_stop_source_sample",
                    "window_start_target_sample",
                    "window_stop_target_sample",
                    "target_sampling_rate_hz",
                    "target_sample_count",
                    "h5_relpath",
                    "events_relpath",
                    "in_libribrain50",
                    "window_complete",
                    "is_trainable",
                    "exclusion_reason",
                ]
            ]
        )

    event_table = pd.concat(tables, ignore_index=True)
    return event_table.sort_values(
        ["session", "onset_seconds", "word_index"]
    ).reset_index(drop=True)


def audit_event_table(event_table, config) -> dict:
    """验证数据划分、计数、通道结构和固定窗口约定。"""
    if event_table.empty:
        raise ValueError("LibriBrain100 事件表为空。")
    split_by_session = _configured_split_by_session(config["split"])
    observed_sessions = set(event_table["session"].astype(int).unique())
    if observed_sessions != set(split_by_session):
        raise ValueError(
            f"事件表 session 与配置不一致：{sorted(observed_sessions)} / "
            f"{sorted(split_by_session)}"
        )
    actual_split = event_table["session"].astype(int).map(split_by_session)
    if not actual_split.eq(event_table["split"]).all():
        raise RuntimeError("发现 session 跨 split 或 split 标签错误。")
    if event_table["event_id"].duplicated().any():
        raise RuntimeError("事件表包含重复 event_id。")
    trainable = event_table[event_table["is_trainable"].map(_parse_bool)]
    if trainable["target_sample_count"].nunique() != 1:
        raise RuntimeError("固定窗口的目标采样点数不唯一。")

    schemas = []
    for relative_path in event_table["h5_relpath"].drop_duplicates():
        schemas.append(
            inspect_h5_schema(Path(config["root"]) / str(relative_path))
        )
    reference = schemas[0]
    for schema in schemas[1:]:
        for key in ("channel_count", "sampling_rate_hz", "channel_names", "channel_types"):
            if schema[key] != reference[key]:
                raise RuntimeError(f"跨 recording 的 H5 {key} 不一致。")

    counts = {
        split: int(trainable["split"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    expected_counts = config.get("expected_trainable_counts")
    if expected_counts:
        expected = {key: int(value) for key, value in expected_counts.items()}
        if counts != expected:
            raise RuntimeError(f"可训练词数漂移：实际 {counts}，预期 {expected}。")
    return {
        "status": "event_table_validated",
        "eeg_signal_loaded": False,
        "recording_count": int(event_table["recording_id"].nunique()),
        "row_count": int(len(event_table)),
        "trainable_counts": counts,
        "channel_count": int(reference["channel_count"]),
        "source_sampling_rate_hz": float(reference["sampling_rate_hz"]),
        "target_sampling_rate_hz": float(trainable["target_sampling_rate_hz"].iloc[0]),
        "target_sample_count": int(trainable["target_sample_count"].iloc[0]),
        "eligibility_window_seconds": float(
            config.get("eligibility_window_seconds", config.get("window_seconds", 3.0))
        ),
        "observed_libribrain50_words": int(
            trainable.loc[trainable["in_libribrain50"].map(_parse_bool), "normalized_word"].nunique()
        ),
        "split_sessions": {
            split: sorted(
                trainable.loc[trainable["split"].eq(split), "session"]
                .astype(int)
                .unique()
                .tolist()
            )
            for split in ("train", "val", "test")
        },
    }


def save_event_table(event_table, path, audit=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    event_table.to_csv(temporary_path, index=False, encoding="utf-8")
    os.replace(temporary_path, path)
    if audit is not None:
        audit_path = path.with_suffix(".audit.json")
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return path


def _parse_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"无法解析布尔值：{value!r}")


def load_event_table(path, split=None, trainable_only=True) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "split",
        "normalized_word",
        "sentence_uid",
        "h5_relpath",
        "window_start_target_sample",
        "is_trainable",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"缓存事件表缺少字段：{missing}")
    if trainable_only:
        table = table[table["is_trainable"].map(_parse_bool)]
    if split is not None:
        table = table[table["split"].eq(split)]
    if table.empty:
        raise ValueError(f"事件表在 split={split!r} 后为空。")
    return table.sort_values(
        ["session", "onset_seconds", "word_index"]
    ).reset_index(drop=True)


def _preprocessing_signature(config, source_path) -> dict:
    stat = Path(source_path).stat()
    return {
        "source_path": str(Path(source_path).resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_sampling_rate_hz": float(config["source_sampling_rate_hz"]),
        "target_sampling_rate_hz": float(config["target_sampling_rate_hz"]),
        "filter_low_hz": config.get("filter_low_hz"),
        "filter_high_hz": config.get("filter_high_hz"),
        "filter_order": int(config.get("filter_order", 4)),
        "scaler": str(config.get("scaler", "RobustScaler")),
    }


def _signature_digest(signature) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def processed_recording_path(source_path, config, cache_dir) -> Path:
    signature = _preprocessing_signature(config, source_path)
    return Path(cache_dir) / f"{Path(source_path).stem}_{_signature_digest(signature)}.npy"


def _valid_recording_cache(cache_path, signature) -> bool:
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return False
        array = np.load(cache_path, mmap_mode="r")
        expected_shape = tuple(metadata["shape"])
        return array.shape == expected_shape and array.dtype == np.float32
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def materialize_recording_cache(source_path, config, cache_dir, force=False) -> Path:
    """对一段完整记录进行滤波、重采样和稳健缩放，然后写入缓存。"""
    from scipy import signal

    source_path = Path(source_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _preprocessing_signature(config, source_path)
    cache_path = processed_recording_path(source_path, config, cache_dir)
    if not force and _valid_recording_cache(cache_path, signature):
        return cache_path

    with h5py.File(source_path, "r") as handle:
        source_rate = float(handle.attrs["sample_frequency"])
        data = np.asarray(handle["data"], dtype=np.float32)
    expected_rate = float(config["source_sampling_rate_hz"])
    if not math.isclose(source_rate, expected_rate):
        raise ValueError(f"{source_path} 的采样率为 {source_rate}，预期 {expected_rate}。")

    low = config.get("filter_low_hz")
    high = config.get("filter_high_hz")
    if low is not None or high is not None:
        if low is None:
            cutoff = float(high)
            kind = "lowpass"
        elif high is None:
            cutoff = float(low)
            kind = "highpass"
        else:
            cutoff = (float(low), float(high))
            kind = "bandpass"
        sos = signal.butter(
            int(config.get("filter_order", 4)),
            cutoff,
            btype=kind,
            fs=source_rate,
            output="sos",
        ).astype(np.float32)
        data = signal.sosfiltfilt(sos, data, axis=1).astype(np.float32, copy=False)

    target_rate = float(config["target_sampling_rate_hz"])
    if not math.isclose(source_rate, target_rate):
        divisor = math.gcd(round(source_rate), round(target_rate))
        up = round(target_rate) // divisor
        down = round(source_rate) // divisor
        data = signal.resample_poly(data, up, down, axis=1).astype(
            np.float32, copy=False
        )

    scaler = str(config.get("scaler", "RobustScaler"))
    if scaler == "RobustScaler":
        quartiles = np.percentile(data, [25.0, 50.0, 75.0], axis=1)
        center = quartiles[1].astype(np.float32)
        scale = (quartiles[2] - quartiles[0]).astype(np.float32)
        scale[scale == 0] = 1.0
        data -= center[:, None]
        data /= scale[:, None]
    elif scaler == "StandardScaler":
        center = data.mean(axis=1, dtype=np.float64).astype(np.float32)
        scale = data.std(axis=1, dtype=np.float64).astype(np.float32)
        scale[scale == 0] = 1.0
        data -= center[:, None]
        data /= scale[:, None]
    elif scaler.lower() not in {"none", "null"}:
        raise ValueError(f"未知 MEG scaler：{scaler}")
    if not np.isfinite(data).all():
        raise RuntimeError(f"预处理后出现非有限值：{source_path}")

    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, data.astype(np.float32, copy=False), allow_pickle=False)
    os.replace(temporary_path, cache_path)
    metadata = {
        "status": "materialized",
        "signature": signature,
        "shape": list(data.shape),
        "dtype": "float32",
    }
    metadata_path = cache_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cache_path


def ensure_recording_caches(event_table, dataset_config, cache_dir, force=False):
    """物化所选事件表引用的每一段记录。"""
    outputs = []
    root = Path(dataset_config["root"])
    paths = event_table["h5_relpath"].drop_duplicates().tolist()
    for index, relative_path in enumerate(paths, start=1):
        source_path = root / str(relative_path)
        print(f"MEG 缓存 {index}/{len(paths)}：{source_path.name}")
        outputs.append(
            materialize_recording_cache(
                source_path, dataset_config, cache_dir, force=force
            )
        )
    return outputs


def _embedding_signature(config) -> dict:
    return {
        "model_name": str(config.get("model_name", "t5-large")),
        "layer_fraction": float(config.get("layer_fraction", 0.5)),
        "token_aggregation": str(config.get("token_aggregation", "mean")),
        "add_special_tokens": False,
        "padding_aggregation": "attention_masked",
    }


def load_text_embedding_cache(path, expected_signature=None) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文本向量缓存不存在：{path}")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"文本向量缓存缺少元数据：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if expected_signature is not None and metadata.get("signature") != expected_signature:
        raise ValueError("文本向量缓存与当前模型/层配置不一致，请使用新的缓存路径。")
    with np.load(path, allow_pickle=False) as payload:
        words = payload["words"].astype(str).tolist()
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if len(words) != len(embeddings) or len(words) != len(set(words)):
        raise ValueError(f"文本向量缓存索引无效：{path}")
    return {word: embeddings[index] for index, word in enumerate(words)}


def ensure_text_embedding_cache(words, config, path) -> Path:
    """增量缓存所选数据划分需要的 T5 词向量。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = _embedding_signature(config)
    existing = {}
    if path.exists():
        existing = load_text_embedding_cache(path, expected_signature=signature)
    requested = sorted({normalize_word(word) for word in words if normalize_word(word)})
    missing = [word for word in requested if word not in existing]
    if not missing:
        return path

    try:
        from transformers import AutoModelForTextEncoding, AutoTokenizer
    except ImportError as exc:
        raise ImportError("生成 T5 词向量需要 transformers。") from exc

    model_name = signature["model_name"]
    local_only = bool(config.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side="left",
        local_files_only=local_only,
    )
    model = AutoModelForTextEncoding.from_pretrained(
        model_name, local_files_only=local_only
    )
    requested_device = str(config.get("device", "cpu"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model.to(device).eval()
    batch_size = int(config.get("batch_size", 32))
    expected_dimension = int(config.get("embedding_dimension", 1024))

    for start in range(0, len(missing), batch_size):
        batch_words = missing[start : start + batch_size]
        print(
            f"T5 词向量 {min(start + len(batch_words), len(missing))}/{len(missing)}"
        )
        inputs = tokenizer(
            batch_words,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        states = outputs.hidden_states
        layer_index = int(signature["layer_fraction"] * len(states) - 1e-6)
        layer_index = min(max(layer_index, 0), len(states) - 1)
        hidden = states[layer_index]
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if signature["token_aggregation"] == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        elif signature["token_aggregation"] == "sum":
            pooled = (hidden * mask).sum(dim=1)
        elif signature["token_aggregation"] == "first":
            pooled = hidden[:, 0]
        elif signature["token_aggregation"] == "last":
            last = inputs["attention_mask"].sum(dim=1).sub(1).clamp_min(0)
            pooled = hidden[torch.arange(len(hidden), device=device), last]
        else:
            raise ValueError(
                f"未知 token aggregation：{signature['token_aggregation']}"
            )
        pooled = pooled.float().cpu().numpy()
        if pooled.shape[1] != expected_dimension:
            raise ValueError(
                f"文本向量维数为 {pooled.shape[1]}，配置预期 {expected_dimension}。"
            )
        for word, embedding in zip(batch_words, pooled):
            existing[word] = embedding.astype(np.float32, copy=False)

    ordered_words = sorted(existing)
    matrix = np.stack([existing[word] for word in ordered_words]).astype(np.float32)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez(file, words=np.asarray(ordered_words), embeddings=matrix)
    os.replace(temporary_path, path)
    metadata = {
        "status": "materialized",
        "signature": signature,
        "word_count": len(ordered_words),
        "embedding_dimension": int(matrix.shape[1]),
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def vectorview_channel_positions(channel_names, layout_path=None) -> np.ndarray:
    """从 MNE 内置布局文件中解析 306 个 Neuromag 通道的位置。"""
    if layout_path is None:
        spec = importlib.util.find_spec("mne")
        if spec is None or spec.origin is None:
            raise ImportError(
                "需要安装 mne，或在配置中提供 Vectorview-all.lout 的 layout_path。"
            )
        layout_path = (
            Path(spec.origin).parent
            / "channels"
            / "data"
            / "layouts"
            / "Vectorview-all.lout"
        )
    layout_path = Path(layout_path)
    if not layout_path.exists():
        raise FileNotFoundError(f"找不到 Vectorview layout：{layout_path}")
    positions = {}
    lines = layout_path.read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        fields = line.split()
        if len(fields) < 7:
            continue
        name = "".join(fields[5:])
        positions[name] = (float(fields[1]), float(fields[2]))
    try:
        output = np.asarray(
            [positions[str(name).replace(" ", "")] for name in channel_names],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise KeyError(f"Vectorview layout 缺少通道 {exc.args[0]}。") from exc
    lower = output.min(axis=0, keepdims=True)
    span = output.max(axis=0, keepdims=True) - lower
    output = (output - lower) / np.maximum(span, 1e-8)
    return output.astype(np.float32)


@lru_cache(maxsize=24)
def _open_processed_recording(path: str):
    return np.load(path, mmap_mode="r")


class LibriBrainWordDataset(Dataset):
    """将固定的 3 秒 MEG 窗口与缓存的 T5 词向量配对。"""

    def __init__(
        self,
        event_table,
        dataset_config,
        meg_cache_dir,
        embedding_cache_path,
        text_embedding_config,
        zero_meg=False,
    ):
        self.table = event_table.reset_index(drop=True).copy()
        self.dataset_config = dict(dataset_config)
        self.meg_cache_dir = Path(meg_cache_dir)
        self.zero_meg = bool(zero_meg)
        self.embedding_map = load_text_embedding_cache(
            embedding_cache_path,
            expected_signature=_embedding_signature(text_embedding_config),
        )
        missing = sorted(
            set(self.table["normalized_word"].astype(str)) - set(self.embedding_map)
        )
        if missing:
            raise ValueError(f"文本向量缓存缺少 {len(missing)} 个词，例如 {missing[:5]}。")

        root = Path(self.dataset_config["root"])
        first_h5 = root / str(self.table["h5_relpath"].iloc[0])
        schema = inspect_h5_schema(first_h5)
        self.channel_names = schema["channel_names"] or tuple(
            f"MEG{index:03d}" for index in range(schema["channel_count"])
        )
        self.channel_positions = vectorview_channel_positions(
            self.channel_names, self.dataset_config.get("layout_path")
        )
        self.channel_count = schema["channel_count"]
        self.window_samples = int(self.table["target_sample_count"].iloc[0])
        self.baseline_samples = round(
            float(self.dataset_config.get("baseline_seconds", 0.5))
            * float(self.dataset_config["target_sampling_rate_hz"])
        )
        self.clamp = self.dataset_config.get("clamp", 5.0)
        sentence_uids = self.table["sentence_uid"].astype(str).tolist()
        sentence_to_index = {
            value: index for index, value in enumerate(dict.fromkeys(sentence_uids))
        }
        self.sentence_indices = np.asarray(
            [sentence_to_index[value] for value in sentence_uids], dtype=np.int64
        )
        self.subject_count = 1
        self.recording_cache_paths = {}
        for relative_path in self.table["h5_relpath"].drop_duplicates():
            source_path = root / str(relative_path)
            cache_path = processed_recording_path(
                source_path, self.dataset_config, self.meg_cache_dir
            )
            signature = _preprocessing_signature(self.dataset_config, source_path)
            if not self.zero_meg and not _valid_recording_cache(cache_path, signature):
                raise FileNotFoundError(
                    f"MEG 缓存尚未物化：{cache_path}。"
                    "请先运行 prepare --with-meg-cache。"
                )
            self.recording_cache_paths[str(relative_path)] = cache_path

    def __len__(self):
        return len(self.table)

    def _recording_path(self, row) -> Path:
        return self.recording_cache_paths[str(row["h5_relpath"])]

    def read_meg(self, index) -> np.ndarray:
        row = self.table.iloc[int(index)]
        if self.zero_meg:
            return np.zeros((self.channel_count, self.window_samples), dtype=np.float32)
        recording = _open_processed_recording(str(self._recording_path(row)))
        start = int(row["window_start_target_sample"])
        stop = start + self.window_samples
        if start < 0 or stop > recording.shape[1]:
            raise IndexError(
                f"窗口越界：{row['event_id']} [{start}, {stop}) / {recording.shape[1]}"
            )
        window = np.array(recording[:, start:stop], dtype=np.float32, copy=True)
        if self.baseline_samples > 0:
            window -= window[:, : self.baseline_samples].mean(axis=1, keepdims=True)
        if self.clamp is not None:
            np.clip(window, -float(self.clamp), float(self.clamp), out=window)
        return window

    def __getitem__(self, index):
        row = self.table.iloc[int(index)]
        word = str(row["normalized_word"])
        return {
            "meg": torch.from_numpy(self.read_meg(index)),
            "text_embedding": torch.from_numpy(
                np.array(self.embedding_map[word], dtype=np.float32, copy=True)
            ),
            "subject_index": torch.tensor(0, dtype=torch.long),
            "sentence_index": torch.tensor(
                self.sentence_indices[int(index)], dtype=torch.long
            ),
            "word": word,
            "event_id": str(row["event_id"]),
            "recording_id": str(row["recording_id"]),
        }
