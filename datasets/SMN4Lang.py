"""SMN4Lang（OpenNeuro ds004078）词事件适配器与缓存工具。

公开词对齐使用 fMRI 时钟。本适配器移除公开数据中的 10.65 秒 fMRI 偏移，
将每个故事锚定到实测 MEG ``Beg`` 事件，并加上文档注明的 39.5 毫秒声学传输
延迟。事件表中的时间相对于各 FIF 保存的第一个采样点，这也是物化后 NumPy
记录所使用的坐标系。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from braindecoding.data.sensors import vectorview_channel_positions
from braindecoding.data.text import (
    ensure_text_embedding_cache,
    load_text_embedding_cache,
    text_embedding_signature,
)
from braindecoding.events import add_core_event_columns


FMRI_ALIGNMENT_OFFSET_SECONDS = 10.65
MEG_AUDIO_DELAY_SECONDS = 0.0395
RECORDING_PATTERN = re.compile(
    r"sub-(?P<subject>\d+)_task-(?P<task>[^_]+)_run-(?P<run>\d+)_meg\.fif$"
)
MEG_CHANNEL_TYPES = frozenset({"MEGMAG", "MEGGRADPLANAR"})
REPOSITORY_GPT2_SOURCE = "smn4lang_repository_gpt2"

# 冻结自 OpenNeuro ds004078 1.2.1 快照中的训练故事 1--50。
# 频次相同时按字典序排列，以保证约定具有确定性。
SMN4LANG50_VOCABULARY = (
    "的",
    "是",
    "一",
    "在",
    "不",
    "了",
    "有",
    "也",
    "人",
    "和",
    "这",
    "个",
    "就",
    "上",
    "更",
    "我们",
    "来",
    "到",
    "能",
    "从",
    "对",
    "与",
    "但",
    "多",
    "中",
    "说",
    "而",
    "让",
    "都",
    "着",
    "生活",
    "种",
    "为",
    "直播",
    "还",
    "会",
    "之",
    "成为",
    "文化",
    "需要",
    "要",
    "等",
    "新",
    "被",
    "年",
    "发展",
    "中国",
    "教育",
    "平台",
    "一些",
)


def normalize_word(value: object) -> str:
    """去除对齐填充，只保留字母数字形式的词内容。"""
    return "".join(
        character
        for character in str(value).strip()
        if character.isalnum() or character in {"-", "'"}
    ).lower()


def training_event_mask(frame: pd.DataFrame) -> pd.Series:
    """选择所有非空且具有完整解码窗口的词。"""
    return frame["normalized_word"].ne("") & frame["window_complete"].eq(True)


def script_sentence_indices(
    words,
    script_path,
    max_character_mismatches=1,
) -> tuple[np.ndarray, dict]:
    """将已对齐的词映射到公开故事脚本中每行一个的句子。

    词时间 MAT 文件不含句子标识，而 ``scripts/story_*.txt`` 中每个非空行都是
    一个完整句子。去除标点后，冻结快照中的两条字符流除一个已知的单字转写差异
    外完全一致。要求字符流等长且汉明距离较小，可以防止一次插入或删除在无提示
    的情况下偏移后续全部句界。
    """
    script_path = Path(script_path)
    sentences = [
        normalize_word(line)
        for line in script_path.read_text(encoding="utf-8-sig").splitlines()
    ]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        raise ValueError(f"No non-empty sentences found in {script_path}.")

    normalized_words = [normalize_word(word) for word in words]
    word_stream = "".join(normalized_words)
    script_stream = "".join(sentences)
    if len(word_stream) != len(script_stream):
        raise ValueError(
            f"Word/script character counts differ for {script_path}: "
            f"{len(word_stream)} != {len(script_stream)}."
        )
    mismatch_count = sum(
        word_character != script_character
        for word_character, script_character in zip(word_stream, script_stream)
    )
    if mismatch_count > int(max_character_mismatches):
        raise ValueError(
            f"Word/script streams differ at {mismatch_count} characters for "
            f"{script_path}; allowed {max_character_mismatches}."
        )

    sentence_stops = np.cumsum([len(sentence) for sentence in sentences])
    sentence_indices = []
    character_cursor = 0
    for word in normalized_words:
        word_start = character_cursor
        character_cursor += len(word)
        sentence_index = int(
            np.searchsorted(sentence_stops, character_cursor, side="left")
        )
        sentence_index = min(sentence_index, len(sentences) - 1)
        sentence_start = 0 if sentence_index == 0 else int(sentence_stops[sentence_index - 1])
        if word and word_start < sentence_start:
            raise ValueError(
                f"An aligned word crosses a script sentence boundary in {script_path}."
            )
        sentence_indices.append(sentence_index)

    return np.asarray(sentence_indices, dtype=np.int64), {
        "sentence_count": len(sentences),
        "normalized_character_count": len(script_stream),
        "character_mismatch_count": mismatch_count,
    }


def split_for_run(run: int) -> str:
    """返回指定运行编号对应的冻结完整故事划分。"""
    run = int(run)
    if 1 <= run <= 50:
        return "train"
    if 51 <= run <= 55:
        return "val"
    if 56 <= run <= 60:
        return "test"
    raise ValueError(f"SMN4Lang run must be in [1, 60], got {run}.")


def meg_word_times(fmri_starts, fmri_stops, meg_begin_seconds):
    """将公开的强制对齐时间转换为相对文件起点的 MEG 时间。"""
    starts = (
        np.asarray(fmri_starts, dtype=float)
        - FMRI_ALIGNMENT_OFFSET_SECONDS
        + float(meg_begin_seconds)
        + MEG_AUDIO_DELAY_SECONDS
    )
    stops = (
        np.asarray(fmri_stops, dtype=float)
        - FMRI_ALIGNMENT_OFFSET_SECONDS
        + float(meg_begin_seconds)
        + MEG_AUDIO_DELAY_SECONDS
    )
    return starts, stops


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _configured_split_by_run(split_config) -> dict[int, str]:
    mapping = {}
    for split_name in ("train", "val", "test"):
        for run in (int(value) for value in split_config.get(f"{split_name}_runs", [])):
            if run in mapping:
                raise ValueError(f"run {run} is assigned to more than one split.")
            mapping[run] = split_name
    if not mapping:
        raise ValueError("No train/val/test runs were configured.")
    return mapping


def _recording_files(dataset_config) -> list[dict]:
    root = Path(dataset_config["root"])
    data_root = root / dataset_config.get(
        "preprocessed_dir", "derivatives/preprocessed_data"
    )
    annotation_root = root / dataset_config.get(
        "word_alignment_dir", "derivatives/annotations/time_align/word-level"
    )
    script_root = root / dataset_config.get(
        "script_dir", "derivatives/annotations/scripts"
    )
    if not data_root.exists() or not annotation_root.exists() or not script_root.exists():
        raise FileNotFoundError(
            f"SMN4Lang requires {data_root}, {annotation_root}, and {script_root}."
        )

    split_by_run = _configured_split_by_run(dataset_config["split"])
    configured_subjects = tuple(
        str(value).removeprefix("sub-")
        for value in dataset_config.get("subjects", ["sub-01"])
    )
    records = []
    for subject in configured_subjects:
        meg_dir = data_root / f"sub-{subject}" / "MEG"
        for run in sorted(split_by_run):
            matches = list(
                meg_dir.glob(f"sub-{subject}_task-RDR_run-{run}_meg.fif")
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one sub-{subject} run-{run} FIF, found {len(matches)}."
                )
            fif_path = matches[0]
            match = RECORDING_PATTERN.match(fif_path.name)
            if match is None:
                raise ValueError(f"Cannot parse SMN4Lang filename: {fif_path.name}")
            stem = fif_path.name[: -len("_meg.fif")]
            records.append(
                {
                    "subject": subject,
                    "task": match.group("task"),
                    "run": run,
                    "split": split_by_run[run],
                    "fif_path": fif_path,
                    "sidecar_path": meg_dir / f"{stem}_meg.json",
                    "channels_path": meg_dir / f"{stem}_channels.tsv",
                    "events_path": meg_dir / f"{stem}_events.tsv",
                    "alignment_path": annotation_root / f"story_{run}_word_time.mat",
                    "script_path": script_root / f"story_{run}.txt",
                }
            )
    return records


def inspect_recording_schema(fif_path) -> dict:
    """读取 BIDS 旁侧文件，但不打开大型 FIF 信号数据。"""
    fif_path = Path(fif_path)
    if not fif_path.exists():
        raise FileNotFoundError(fif_path)
    stem = fif_path.name[: -len("_meg.fif")]
    sidecar_path = fif_path.with_name(f"{stem}_meg.json")
    channels_path = fif_path.with_name(f"{stem}_channels.tsv")
    if not sidecar_path.exists() or not channels_path.exists():
        raise FileNotFoundError(
            f"Missing sidecar or channels table for {fif_path}."
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8-sig"))
    channels = pd.read_csv(channels_path, sep="\t", encoding="utf-8-sig")
    required = {"name", "type", "status"}
    missing = sorted(required - set(channels.columns))
    if missing:
        raise ValueError(f"{channels_path} is missing columns {missing}.")
    selected = channels[
        channels["type"].isin(MEG_CHANNEL_TYPES)
        & channels["status"].astype(str).str.lower().eq("good")
    ]
    if selected.empty:
        raise ValueError(f"No good MEG channels found in {channels_path}.")
    sampling_rate = float(sidecar["SamplingFrequency"])
    duration = float(sidecar["RecordingDuration"])
    return {
        "channel_count": int(len(selected)),
        "channel_names": tuple(selected["name"].astype(str)),
        "channel_types": tuple(selected["type"].astype(str)),
        "sampling_rate_hz": sampling_rate,
        "recording_duration_seconds": duration,
        "sample_count": int(round(duration * sampling_rate)),
    }


def _load_alignment(path):
    alignment = loadmat(path, squeeze_me=True, struct_as_record=False)
    starts = np.atleast_1d(alignment["start"]).astype(float)
    stops = np.atleast_1d(alignment["end"]).astype(float)
    words = np.atleast_1d(alignment["word"])
    if not (len(words) == len(starts) == len(stops)):
        raise ValueError(
            f"Alignment field lengths disagree in {path}: "
            f"word={len(words)}, start={len(starts)}, end={len(stops)}."
        )
    return words, starts, stops


def _training_vocabulary(records, size=50) -> tuple[str, ...]:
    counts = Counter()
    seen_runs = set()
    for record in records:
        if record["split"] != "train" or record["run"] in seen_runs:
            continue
        seen_runs.add(record["run"])
        words, _, _ = _load_alignment(record["alignment_path"])
        counts.update(word for word in map(normalize_word, words) if word)
    ranked = sorted(counts, key=lambda word: (-counts[word], word))
    return tuple(ranked[: int(size)])


def uses_repository_gpt2(text_embedding_config) -> bool:
    """Return whether text targets come from the dataset-provided GPT-2 files."""
    return str(text_embedding_config.get("source", "")).strip().lower() == (
        REPOSITORY_GPT2_SOURCE
    )


def _repository_gpt2_paths(dataset_config, text_embedding_config) -> dict[int, Path]:
    root = Path(dataset_config["root"])
    relative_dir = Path(
        text_embedding_config.get(
            "relative_dir", "derivatives/annotations/embeddings/gpt/word-level"
        )
    )
    pattern = str(
        text_embedding_config.get(
            "filename_pattern", "story_{run}_word_gpt_0-24_1024.mat"
        )
    )
    runs = sorted(int(run) for run in dataset_config["split"]["train_runs"])
    paths = {run: root / relative_dir / pattern.format(run=run) for run in runs}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} dataset-provided GPT-2 files: {missing[:3]}"
        )
    return paths


def repository_gpt2_embedding_signature(
    dataset_config, text_embedding_config
) -> dict:
    """Describe the frozen train-only GPT-2 word-prototype contract."""
    paths = _repository_gpt2_paths(dataset_config, text_embedding_config)
    fingerprint = hashlib.sha256()
    for run, path in paths.items():
        stat = path.stat()
        fingerprint.update(
            f"{run}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode(
                "utf-8"
            )
        )
    return {
        "source": REPOSITORY_GPT2_SOURCE,
        "dataset_snapshot": str(dataset_config.get("snapshot", "")),
        "relative_dir": str(text_embedding_config["relative_dir"]),
        "filename_pattern": str(text_embedding_config["filename_pattern"]),
        "layer_index": int(text_embedding_config["layer_index"]),
        "layer_count": int(text_embedding_config.get("layer_count", 25)),
        "embedding_dimension": int(text_embedding_config["embedding_dimension"]),
        "aggregation": str(text_embedding_config["aggregation"]),
        "source_file_count": len(paths),
        "source_files_fingerprint": fingerprint.hexdigest(),
    }


def _training_event_digest(train_table) -> str:
    ordered = train_table.sort_values(["run", "word_index"])
    payload = "\n".join(
        f"{int(row.run)}|{int(row.word_index)}|{row.normalized_word}"
        for row in ordered.itertuples()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_repository_gpt2_word_prototypes(
    event_table,
    dataset_config,
    text_embedding_config,
    cache_path,
) -> Path:
    """Average official contextual GPT-2 vectors into train-only word prototypes."""
    if not uses_repository_gpt2(text_embedding_config):
        raise ValueError("The text embedding config does not select repository GPT-2.")
    required = {"split", "is_trainable", "run", "word_index", "normalized_word"}
    missing_columns = sorted(required - set(event_table.columns))
    if missing_columns:
        raise ValueError(f"Event table is missing GPT-2 alignment fields: {missing_columns}")
    train_mask = event_table["split"].eq("train") & event_table["is_trainable"].map(
        _parse_bool
    )
    train_table = event_table.loc[
        train_mask, ["run", "word_index", "normalized_word"]
    ].copy()
    if train_table.empty:
        cache_path = Path(cache_path)
        if cache_path.exists():
            load_text_embedding_cache(
                cache_path,
                expected_signature=repository_gpt2_embedding_signature(
                    dataset_config, text_embedding_config
                ),
            )
            return cache_path
        raise ValueError(
            "Building repository GPT-2 prototypes requires train rows. "
            "Run prepare --with-text-embeddings first."
        )
    conflicts = (
        train_table.groupby(["run", "word_index"])["normalized_word"].nunique()
    )
    if int(conflicts.max()) != 1:
        raise RuntimeError("A GPT-2 word position maps to multiple normalized words.")
    train_table = train_table.drop_duplicates(["run", "word_index"])
    signature = repository_gpt2_embedding_signature(
        dataset_config, text_embedding_config
    )
    event_digest = _training_event_digest(train_table)
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    expected_words = sorted(train_table["normalized_word"].astype(str).unique())
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing = load_text_embedding_cache(
            cache_path, expected_signature=signature
        )
        if (
            metadata.get("training_event_digest") == event_digest
            and sorted(existing) == expected_words
        ):
            return cache_path
        # Smoke/debug runs use a prefix of the full train table. A complete
        # cache is valid for such a subset and must never be replaced by the
        # smaller selection.
        cached_event_count = int(metadata.get("training_event_count", 0))
        if (
            cached_event_count > len(train_table)
            and set(expected_words).issubset(existing)
        ):
            return cache_path

    layer_index = int(text_embedding_config["layer_index"])
    layer_count = int(text_embedding_config.get("layer_count", 25))
    dimension = int(text_embedding_config["embedding_dimension"])
    if not 0 <= layer_index < layer_count:
        raise ValueError(
            f"GPT-2 layer_index={layer_index} is outside [0, {layer_count})."
        )
    word_to_index = {word: index for index, word in enumerate(expected_words)}
    sums = np.zeros((len(expected_words), dimension), dtype=np.float64)
    counts = np.zeros(len(expected_words), dtype=np.int64)
    paths = _repository_gpt2_paths(dataset_config, text_embedding_config)
    for completed, (run, source_path) in enumerate(paths.items(), start=1):
        rows = train_table[train_table["run"].astype(int).eq(run)]
        if rows.empty:
            continue
        print(f"GPT-2 prototype source {completed}/{len(paths)}: {source_path.name}")
        payload = loadmat(source_path, variable_names=["data"])
        if "data" not in payload:
            raise ValueError(f"GPT-2 MAT file has no data variable: {source_path}")
        data = np.asarray(payload["data"])
        if data.ndim != 3 or data.shape[0] != layer_count or data.shape[2] != dimension:
            raise ValueError(
                f"Unexpected GPT-2 shape {data.shape} in {source_path}; "
                f"expected ({layer_count}, words, {dimension})."
            )
        word_indices = rows["word_index"].to_numpy(dtype=np.int64)
        if word_indices.min() < 0 or word_indices.max() >= data.shape[1]:
            raise IndexError(f"GPT-2 word index exceeds {source_path}: {data.shape}.")
        vectors = np.asarray(data[layer_index, word_indices, :], dtype=np.float64)
        if not np.isfinite(vectors).all():
            raise ValueError(f"GPT-2 source contains non-finite values: {source_path}")
        codes = np.asarray(
            [word_to_index[word] for word in rows["normalized_word"].astype(str)],
            dtype=np.int64,
        )
        np.add.at(sums, codes, vectors)
        np.add.at(counts, codes, 1)
    if np.any(counts == 0):
        raise RuntimeError("At least one training word has no GPT-2 source vector.")
    prototypes = (sums / counts[:, None]).astype(np.float32)
    if not np.isfinite(prototypes).all():
        raise RuntimeError("Computed GPT-2 prototypes contain non-finite values.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            words=np.asarray(expected_words),
            embeddings=prototypes,
            counts=counts,
        )
    os.replace(temporary_path, cache_path)
    metadata = {
        "status": "materialized",
        "signature": signature,
        "training_event_digest": event_digest,
        "training_event_count": int(len(train_table)),
        "word_count": len(expected_words),
        "embedding_dimension": dimension,
        "candidate_provenance": "train_runs_only",
    }
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_metadata, metadata_path)
    return cache_path


def ensure_configured_text_embedding_cache(
    event_table,
    dataset_config,
    text_embedding_config,
    cache_path,
) -> Path:
    """Materialize either repository GPT-2 prototypes or generated word vectors."""
    if uses_repository_gpt2(text_embedding_config):
        return ensure_repository_gpt2_word_prototypes(
            event_table,
            dataset_config,
            text_embedding_config,
            cache_path,
        )
    return ensure_text_embedding_cache(
        event_table["normalized_word"], text_embedding_config, cache_path
    )


def build_event_table(config) -> pd.DataFrame:
    """为每个词构建一行可审计记录，不打开 MEG 信号数据。"""
    records = _recording_files(config)
    vocabulary_size = int(config.get("vocabulary_size", 50))
    vocabulary = _training_vocabulary(records, vocabulary_size)
    if vocabulary_size == 50 and vocabulary != SMN4LANG50_VOCABULARY:
        raise RuntimeError(
            "The train-only SMN4Lang50 vocabulary differs from the frozen "
            "OpenNeuro 1.2.1 contract."
        )

    expected_source_rate = float(config.get("source_sampling_rate_hz", 1000.0))
    target_rate = float(config.get("target_sampling_rate_hz", 50.0))
    window_seconds = float(config.get("window_seconds", 3.0))
    eligibility_window_seconds = float(
        config.get("eligibility_window_seconds", window_seconds)
    )
    if eligibility_window_seconds < window_seconds:
        raise ValueError(
            "eligibility_window_seconds cannot be shorter than window_seconds."
        )
    root = Path(config["root"])
    tables = []
    for record in records:
        for path in (
            record["sidecar_path"],
            record["channels_path"],
            record["events_path"],
            record["alignment_path"],
            record["script_path"],
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing SMN4Lang run file: {path}")
        schema = inspect_recording_schema(record["fif_path"])
        source_rate = schema["sampling_rate_hz"]
        if not math.isclose(source_rate, expected_source_rate):
            raise ValueError(
                f"{record['fif_path']} is {source_rate} Hz; expected "
                f"{expected_source_rate} Hz."
            )
        raw_events = pd.read_csv(
            record["events_path"], sep="\t", encoding="utf-8-sig"
        )
        begins = raw_events.loc[raw_events["trial_type"].eq("Beg"), "onset"]
        ends = raw_events.loc[raw_events["trial_type"].eq("End"), "onset"]
        if len(begins) != 1 or len(ends) != 1:
            raise ValueError(
                f"Expected one Beg and End in {record['events_path']}; "
                f"got {len(begins)} and {len(ends)}."
            )
        support_start = float(begins.iloc[0])
        support_stop = float(ends.iloc[0])
        duration = schema["recording_duration_seconds"]
        if not 0 <= support_start < support_stop <= duration + 1e-3:
            raise ValueError(
                f"Invalid trigger support [{support_start}, {support_stop}] "
                f"for duration {duration}."
            )

        raw_words, fmri_starts, fmri_stops = _load_alignment(
            record["alignment_path"]
        )
        starts, stops = meg_word_times(fmri_starts, fmri_stops, support_start)
        if np.any(stops < starts):
            raise ValueError(f"Negative word duration in {record['alignment_path']}.")
        recording_id = (
            f"sub-{record['subject']}_task-{record['task']}_run-{record['run']}"
        )
        normalized_words = [normalize_word(value) for value in raw_words]
        sentence_indices, script_alignment = script_sentence_indices(
            raw_words,
            record["script_path"],
            max_character_mismatches=int(
                config.get("script_alignment_max_character_mismatches", 1)
            ),
        )
        ranks = {word: index + 1 for index, word in enumerate(vocabulary)}
        frame = pd.DataFrame(
            {
                "dataset": "SMN4Lang",
                "subject_id": f"sub-{record['subject']}",
                "task": record["task"],
                "run": record["run"],
                "recording_id": recording_id,
                "split": record["split"],
                "sentence_index": sentence_indices,
                "word_index": np.arange(len(raw_words), dtype=int),
                "word": [str(value).strip() for value in raw_words],
                "normalized_word": normalized_words,
                "onset_seconds": starts,
                "word_duration_seconds": stops - starts,
            }
        )
        frame["word_id"] = frame["normalized_word"].map(_sha256_text)
        frame["event_id"] = [
            _sha256_text(f"{recording_id}|{index}|{word}|{onset:.6f}")
            for index, word, onset in zip(
                frame["word_index"], frame["normalized_word"], frame["onset_seconds"]
            )
        ]
        frame["source_sampling_rate_hz"] = source_rate
        frame["source_sample_count"] = schema["sample_count"]
        frame["recording_duration_seconds"] = duration
        frame["support_start_seconds"] = support_start
        frame["support_stop_seconds"] = support_stop
        frame["window_start_seconds"] = frame["onset_seconds"]
        frame["window_stop_seconds"] = frame["onset_seconds"] + window_seconds
        frame["eligibility_window_stop_seconds"] = (
            frame["onset_seconds"] + eligibility_window_seconds
        )
        frame["window_start_source_sample"] = np.rint(
            frame["window_start_seconds"] * source_rate
        ).astype(int)
        frame["window_stop_source_sample"] = (
            frame["window_start_source_sample"] + round(window_seconds * source_rate)
        )
        frame["window_start_target_sample"] = np.rint(
            frame["window_start_seconds"] * target_rate
        ).astype(int)
        frame["window_stop_target_sample"] = (
            frame["window_start_target_sample"] + round(window_seconds * target_rate)
        )
        frame["target_sampling_rate_hz"] = target_rate
        frame["target_sample_count"] = round(window_seconds * target_rate)
        frame["fif_relpath"] = record["fif_path"].relative_to(root).as_posix()
        frame["events_relpath"] = record["events_path"].relative_to(root).as_posix()
        frame["alignment_relpath"] = record["alignment_path"].relative_to(root).as_posix()
        frame["script_relpath"] = record["script_path"].relative_to(root).as_posix()
        frame["in_smn4lang50"] = frame["normalized_word"].isin(vocabulary)
        frame["vocabulary_rank"] = frame["normalized_word"].map(ranks)
        frame["window_complete"] = (
            frame["window_start_seconds"].ge(support_start - 1e-9)
            & frame["eligibility_window_stop_seconds"].le(support_stop + 1e-9)
        )
        # 与 LibriBrain 保持一致：监督所有可用词事件。仅由训练集冻结的前 50 词表
        # 只作为评估候选集合。
        frame["is_trainable"] = training_event_mask(frame)
        max_context_words = int(config.get("max_context_words", 128))
        if max_context_words <= 0:
            raise ValueError("max_context_words must be positive.")
        frame["source_sentence_uid"] = frame["sentence_index"].map(
            lambda value: f"{recording_id}_sentence-{int(value)}"
        )
        trainable_ordinal = (
            frame["is_trainable"]
            .astype(int)
            .groupby(frame["source_sentence_uid"], sort=False)
            .cumsum()
            .sub(1)
            .clip(lower=0)
        )
        frame["context_chunk_index"] = (
            trainable_ordinal // max_context_words
        ).astype(int)
        frame["sentence_uid"] = [
            f"{source_uid}_chunk-{chunk_index}"
            for source_uid, chunk_index in zip(
                frame["source_sentence_uid"], frame["context_chunk_index"]
            )
        ]
        frame["context_grouping"] = "script_sentence_then_contiguous_chunks"
        frame["script_character_mismatch_count"] = script_alignment[
            "character_mismatch_count"
        ]
        frame["exclusion_reason"] = np.select(
            [
                frame["normalized_word"].eq(""),
                ~frame["window_complete"],
            ],
            ["empty_normalized_word", "outside_trigger_support"],
            default="",
        )
        frame["timing_status"] = "forced_alignment_transformed_to_meg_clock"
        frame["timing_source_clock"] = "fmri_scan"
        frame["timing_fmri_offset_removed_seconds"] = FMRI_ALIGNMENT_OFFSET_SECONDS
        frame["timing_meg_audio_delay_added_seconds"] = MEG_AUDIO_DELAY_SECONDS
        tables.append(frame)

    event_table = pd.concat(tables, ignore_index=True)
    event_table = event_table.sort_values(
        ["subject_id", "run", "onset_seconds", "word_index"]
    ).reset_index(drop=True)
    return add_event_contract(event_table)


def add_event_contract(event_table: pd.DataFrame) -> pd.DataFrame:
    """保持现有稳定排序，追加 SMN4Lang 中文公共字段。"""
    material_ids = event_table["run"].map(
        lambda run: f"SMN4Lang|story-{int(run):02d}"
    )
    return add_core_event_columns(
        event_table,
        material_ids=material_ids,
        split_units=material_ids,
        start_times=event_table["onset_seconds"],
        end_times=event_table["onset_seconds"]
        + event_table["word_duration_seconds"],
    )


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
    raise ValueError(f"Cannot parse boolean value {value!r}.")


def audit_event_table(event_table, config) -> dict:
    """验证划分隔离、词表来源、通道和时间约定。"""
    if event_table.empty:
        raise ValueError("The SMN4Lang event table is empty.")
    split_by_run = _configured_split_by_run(config["split"])
    if set(event_table["run"].astype(int).unique()) != set(split_by_run):
        raise RuntimeError("Observed runs differ from the configured split contract.")
    expected_split = event_table["run"].astype(int).map(split_by_run)
    if not expected_split.eq(event_table["split"]).all():
        raise RuntimeError("A run crosses the frozen complete-story split.")
    if event_table["event_id"].duplicated().any():
        raise RuntimeError("The event table contains duplicate event IDs.")

    trainable = event_table[event_table["is_trainable"].map(_parse_bool)]
    if trainable["target_sample_count"].nunique() != 1:
        raise RuntimeError("Trainable windows do not have a fixed sample count.")
    expected_grouping = "script_sentence_then_contiguous_chunks"
    if set(event_table["context_grouping"].astype(str)) != {expected_grouping}:
        raise RuntimeError("The event table does not use script-sentence context groups.")
    if (
        event_table.groupby("sentence_uid")["recording_id"].nunique().max()
        != 1
    ):
        raise RuntimeError("A context group crosses recording boundaries.")
    max_context_words = int(config.get("max_context_words", 128))
    context_group_sizes = trainable.groupby("sentence_uid").size()
    if context_group_sizes.empty or int(context_group_sizes.max()) > max_context_words:
        raise RuntimeError(
            f"A context group exceeds max_context_words={max_context_words}."
        )
    vocabulary = tuple(
        event_table.loc[event_table["in_smn4lang50"].map(_parse_bool)]
        .dropna(subset=["vocabulary_rank"])
        .sort_values("vocabulary_rank")
        .drop_duplicates("normalized_word")["normalized_word"]
        .astype(str)
    )
    if vocabulary != SMN4LANG50_VOCABULARY:
        raise RuntimeError(
            "The event-table evaluation vocabulary is not the frozen train-only list."
        )

    root = Path(config["root"])
    schemas = [
        inspect_recording_schema(root / relative_path)
        for relative_path in event_table["fif_relpath"].drop_duplicates()
    ]
    reference = schemas[0]
    for schema in schemas[1:]:
        for key in ("channel_count", "channel_names", "channel_types", "sampling_rate_hz"):
            if schema[key] != reference[key]:
                raise RuntimeError(f"Recording schemas disagree on {key}.")

    annotated_counts = {
        split: int(
            (
                event_table["split"].eq(split)
                & event_table["normalized_word"].ne("")
            ).sum()
        )
        for split in ("train", "val", "test")
    }
    trainable_counts = {
        split: int(trainable["split"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    trainable_unique_word_counts = {
        split: int(
            trainable.loc[trainable["split"].eq(split), "normalized_word"].nunique()
        )
        for split in ("train", "val", "test")
    }
    expected_annotated = config.get("expected_annotated_word_counts")
    if expected_annotated and annotated_counts != {
        key: int(value) for key, value in expected_annotated.items()
    }:
        raise RuntimeError(
            f"Annotated word counts drifted: {annotated_counts} != {expected_annotated}."
        )
    expected_trainable = config.get("expected_trainable_counts")
    if expected_trainable and trainable_counts != {
        key: int(value) for key, value in expected_trainable.items()
    }:
        raise RuntimeError(
            f"Trainable word counts drifted: {trainable_counts} != {expected_trainable}."
        )
    return {
        "status": "event_table_validated",
        "meg_signal_loaded": False,
        "dataset_snapshot": "OpenNeuro ds004078 1.2.1",
        "recording_count": int(event_table["recording_id"].nunique()),
        "subject_count": int(event_table["subject_id"].nunique()),
        "row_count": int(len(event_table)),
        "annotated_word_counts": annotated_counts,
        "trainable_counts": trainable_counts,
        "trainable_unique_word_counts": trainable_unique_word_counts,
        "training_vocabulary_policy": "all_nonempty_complete_window_words",
        "evaluation_vocabulary_policy": "frozen_train_only_top50",
        "channel_count": int(reference["channel_count"]),
        "source_sampling_rate_hz": float(reference["sampling_rate_hz"]),
        "target_sampling_rate_hz": float(trainable["target_sampling_rate_hz"].iloc[0]),
        "target_sample_count": int(trainable["target_sample_count"].iloc[0]),
        "eligibility_window_seconds": float(
            config.get("eligibility_window_seconds", config.get("window_seconds", 3.0))
        ),
        "evaluation_vocabulary_size": len(vocabulary),
        "evaluation_vocabulary": list(vocabulary),
        "evaluation_vocabulary_selected_from": "train_runs_only",
        # 为旧版审计读取器保留兼容别名。这些字段现在只描述评估候选，
        # 不描述监督集合。
        "vocabulary_size": len(vocabulary),
        "vocabulary": list(vocabulary),
        "vocabulary_selected_from": "train_runs_only",
        "test_meg_used_for_vocabulary_selection": False,
        "context_contract": {
            "grouping": expected_grouping,
            "max_context_words": max_context_words,
            "trainable_group_count": int(context_group_sizes.size),
            "largest_trainable_group": int(context_group_sizes.max()),
            "script_character_mismatches": int(
                event_table.drop_duplicates("recording_id")[
                    "script_character_mismatch_count"
                ].sum()
            ),
        },
        "timing_contract": {
            "status": "forced_alignment_transformed_to_meg_clock",
            "fmri_offset_removed_seconds": FMRI_ALIGNMENT_OFFSET_SECONDS,
            "meg_audio_delay_added_seconds": MEG_AUDIO_DELAY_SECONDS,
            "event_coordinates": "relative_to_first_stored_fif_sample",
        },
        "split_runs": {
            split: sorted(
                trainable.loc[trainable["split"].eq(split), "run"]
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
        path.with_suffix(".audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return path


def load_event_table(path, split=None, trainable_only=True) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "split",
        "normalized_word",
        "sentence_uid",
        "source_sentence_uid",
        "context_chunk_index",
        "context_grouping",
        "fif_relpath",
        "window_start_target_sample",
        "is_trainable",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Cached event table is missing columns {missing}.")
    if trainable_only:
        table = table[table["is_trainable"].map(_parse_bool)]
    if split is not None:
        table = table[table["split"].eq(split)]
    if table.empty:
        raise ValueError(f"The event table is empty after split={split!r}.")
    return table.sort_values(
        ["subject_id", "run", "onset_seconds", "word_index"]
    ).reset_index(drop=True)


def _preprocessing_signature(config, source_path) -> dict:
    source_path = Path(source_path)
    stat = source_path.stat()
    schema = inspect_recording_schema(source_path)
    return {
        "source_path": str(source_path.resolve()),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_sampling_rate_hz": float(config["source_sampling_rate_hz"]),
        "target_sampling_rate_hz": float(config["target_sampling_rate_hz"]),
        "filter_low_hz": config.get("filter_low_hz"),
        "filter_high_hz": config.get("filter_high_hz"),
        "filter_order": int(config.get("filter_order", 4)),
        "scaler": str(config.get("scaler", "RobustScaler")),
        "channel_names": list(schema["channel_names"]),
    }


def _signature_digest(signature) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def processed_recording_path(source_path, config, cache_dir) -> Path:
    signature = _preprocessing_signature(config, source_path)
    return Path(cache_dir) / f"{Path(source_path).stem}_{_signature_digest(signature)}.npy"


def _valid_recording_cache(cache_path, signature) -> bool:
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return False
        array = np.load(cache_path, mmap_mode="r")
        return (
            array.shape == tuple(metadata["shape"])
            and array.dtype == np.float32
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _read_raw_fif(path):
    import mne

    return mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")


def _scale_channels(data, scaler):
    if scaler == "RobustScaler":
        quartiles = np.percentile(data, [25.0, 50.0, 75.0], axis=1)
        center = quartiles[1].astype(np.float32)
        scale = (quartiles[2] - quartiles[0]).astype(np.float32)
    elif scaler == "StandardScaler":
        center = data.mean(axis=1, dtype=np.float64).astype(np.float32)
        scale = data.std(axis=1, dtype=np.float64).astype(np.float32)
    elif scaler.lower() in {"none", "null"}:
        return data
    else:
        raise ValueError(f"Unknown MEG scaler: {scaler}")
    scale[scale == 0] = 1.0
    data -= center[:, None]
    data /= scale[:, None]
    return data


def materialize_recording_cache(source_path, config, cache_dir, force=False) -> Path:
    """读取一段 FIF 记录，并按需滤波、降采样和缩放。"""
    from scipy import signal

    source_path = Path(source_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _preprocessing_signature(config, source_path)
    cache_path = processed_recording_path(source_path, config, cache_dir)
    if not force and _valid_recording_cache(cache_path, signature):
        return cache_path

    raw = _read_raw_fif(source_path)
    try:
        source_rate = float(raw.info["sfreq"])
        expected_rate = float(config["source_sampling_rate_hz"])
        if not math.isclose(source_rate, expected_rate):
            raise ValueError(
                f"{source_path} is {source_rate} Hz; expected {expected_rate} Hz."
            )
        channel_names = tuple(signature["channel_names"])
        missing = sorted(set(channel_names) - set(raw.ch_names))
        if missing:
            raise ValueError(f"FIF is missing configured channels: {missing[:5]}")
        target_rate = float(config["target_sampling_rate_hz"])
        divisor = math.gcd(round(source_rate), round(target_rate))
        up = round(target_rate) // divisor
        down = round(source_rate) // divisor
        low = config.get("filter_low_hz")
        high = config.get("filter_high_hz")
        sos = None
        if low is not None or high is not None:
            if low is None:
                cutoff, kind = float(high), "lowpass"
            elif high is None:
                cutoff, kind = float(low), "highpass"
            else:
                cutoff, kind = (float(low), float(high)), "bandpass"
            sos = signal.butter(
                int(config.get("filter_order", 4)),
                cutoff,
                btype=kind,
                fs=source_rate,
                output="sos",
            ).astype(np.float32)

        chunks = []
        chunk_size = int(config.get("materialization_channel_chunk", 24))
        scaler = str(config.get("scaler", "RobustScaler"))
        for start in range(0, len(channel_names), chunk_size):
            names = channel_names[start : start + chunk_size]
            data = raw.get_data(picks=list(names)).astype(np.float32)
            if sos is not None:
                data = signal.sosfiltfilt(sos, data, axis=1).astype(
                    np.float32, copy=False
                )
            if not math.isclose(source_rate, target_rate):
                data = signal.resample_poly(data, up, down, axis=1).astype(
                    np.float32, copy=False
                )
            chunks.append(_scale_channels(data, scaler))
        output = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    finally:
        raw.close()
    if not np.isfinite(output).all():
        raise RuntimeError(f"Non-finite values after preprocessing {source_path}.")

    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, output, allow_pickle=False)
    os.replace(temporary_path, cache_path)
    cache_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": "materialized",
                "signature": signature,
                "shape": list(output.shape),
                "dtype": "float32",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return cache_path


def ensure_recording_caches(event_table, dataset_config, cache_dir, force=False):
    outputs = []
    root = Path(dataset_config["root"])
    paths = event_table["fif_relpath"].drop_duplicates().tolist()
    for index, relative_path in enumerate(paths, start=1):
        source_path = root / str(relative_path)
        print(f"MEG cache {index}/{len(paths)}: {source_path.name}")
        outputs.append(
            materialize_recording_cache(
                source_path, dataset_config, cache_dir, force=force
            )
        )
    return outputs


@lru_cache(maxsize=24)
def _open_processed_recording(path: str):
    return np.load(path, mmap_mode="r")


class SMN4LangWordDataset(Dataset):
    """将词起点后的固定 MEG 窗口与配置的词原型配对。"""

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
        self.uses_repository_gpt2 = uses_repository_gpt2(text_embedding_config)
        if self.uses_repository_gpt2:
            expected_signature = repository_gpt2_embedding_signature(
                self.dataset_config, text_embedding_config
            )
        else:
            expected_signature = text_embedding_signature(text_embedding_config)
        self.embedding_map = load_text_embedding_cache(
            embedding_cache_path,
            expected_signature=expected_signature,
        )
        missing = sorted(
            set(self.table["normalized_word"].astype(str)) - set(self.embedding_map)
        )
        if self.uses_repository_gpt2:
            required_mask = self.table["split"].eq("train")
            if "in_smn4lang50" in self.table:
                required_mask |= self.table["in_smn4lang50"].map(_parse_bool)
            required_words = set(
                self.table.loc[required_mask, "normalized_word"].astype(str)
            )
            missing_required = sorted(required_words - set(self.embedding_map))
            if missing_required:
                raise ValueError(
                    "Train or fixed-vocabulary GPT-2 prototypes are missing: "
                    f"{missing_required[:5]}"
                )
            self.missing_embedding = np.zeros(
                int(text_embedding_config["embedding_dimension"]), dtype=np.float32
            )
        elif missing:
            raise ValueError(
                f"Text embedding cache is missing {len(missing)} words: {missing[:5]}"
            )
        else:
            self.missing_embedding = None
        root = Path(self.dataset_config["root"])
        first_source = root / str(self.table["fif_relpath"].iloc[0])
        schema = inspect_recording_schema(first_source)
        self.channel_names = schema["channel_names"]
        self.channel_positions = vectorview_channel_positions(
            self.channel_names, self.dataset_config.get("layout_path")
        )
        self.channel_count = schema["channel_count"]
        self.window_samples = int(self.table["target_sample_count"].iloc[0])
        self.baseline_samples = round(
            float(self.dataset_config.get("baseline_seconds", 0.0))
            * float(self.dataset_config["target_sampling_rate_hz"])
        )
        self.clamp = self.dataset_config.get("clamp", 5.0)
        sentence_uids = self.table["sentence_uid"].astype(str).tolist()
        sentence_map = {
            value: index for index, value in enumerate(dict.fromkeys(sentence_uids))
        }
        self.sentence_indices = np.asarray(
            [sentence_map[value] for value in sentence_uids], dtype=np.int64
        )
        subject_ids = self.table["subject_id"].astype(str).tolist()
        self.subject_map = {
            value: index for index, value in enumerate(dict.fromkeys(subject_ids))
        }
        self.subject_indices = np.asarray(
            [self.subject_map[value] for value in subject_ids], dtype=np.int64
        )
        self.subject_count = len(self.subject_map)
        self.recording_cache_paths = {}
        for relative_path in self.table["fif_relpath"].drop_duplicates():
            source_path = root / str(relative_path)
            cache_path = processed_recording_path(
                source_path, self.dataset_config, self.meg_cache_dir
            )
            signature = _preprocessing_signature(self.dataset_config, source_path)
            if not self.zero_meg and not _valid_recording_cache(cache_path, signature):
                raise FileNotFoundError(
                    f"MEG cache has not been materialized: {cache_path}. "
                    "Run prepare --with-meg-cache first."
                )
            self.recording_cache_paths[str(relative_path)] = cache_path

    def __len__(self):
        return len(self.table)

    def read_meg(self, index) -> np.ndarray:
        row = self.table.iloc[int(index)]
        if self.zero_meg:
            return np.zeros(
                (self.channel_count, self.window_samples), dtype=np.float32
            )
        path = self.recording_cache_paths[str(row["fif_relpath"])]
        recording = _open_processed_recording(str(path))
        start = int(row["window_start_target_sample"])
        stop = start + self.window_samples
        if start < 0 or stop > recording.shape[1]:
            raise IndexError(
                f"Window {row['event_id']} [{start}, {stop}) exceeds "
                f"recording length {recording.shape[1]}."
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
        embedding = self.embedding_map.get(word, self.missing_embedding)
        if embedding is None:
            raise KeyError(f"No configured text embedding for {word!r}.")
        return {
            "meg": torch.from_numpy(self.read_meg(index)),
            "text_embedding": torch.from_numpy(
                np.array(embedding, dtype=np.float32, copy=True)
            ),
            "subject_index": torch.tensor(
                self.subject_indices[int(index)], dtype=torch.long
            ),
            "sentence_index": torch.tensor(
                self.sentence_indices[int(index)], dtype=torch.long
            ),
            "word": word,
            "event_id": str(row["event_id"]),
            "recording_id": str(row["recording_id"]),
        }
