"""ChineseEEG 静默阅读行级数据集。建表阶段不加载 EEG 信号。"""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


NOVELS = ("GarnettDream", "LittlePrince")
EVENT_FILE_PATTERN = re.compile(
    r"sub-(?P<subject>\d+)_ses-(?P<novel>GarnettDream|LittlePrince)_"
    r"task-reading_run-(?P<run>\d+)_events\.tsv$"
)

# 与刺激呈现程序一致：这些符号不参与高亮字符计数。
PRESENTATION_PUNCTUATION = frozenset(
    ["\n", "。", "，", "！", "？", "：", "；", "“", "”", "、", "《", "》", ".", "（", "）", "…", "·"]
)


def select_novels(novel):
    """根据配置选择《狼王梦》《小王子》或全部小说。"""
    names = {name.lower(): name for name in NOVELS}
    selected = str(novel).lower()
    if selected in {"all", "全部"}:
        return NOVELS
    if selected not in names:
        raise ValueError("novel 只能是 GarnettDream、LittlePrince 或 all。")
    return (names[selected],)


def find_event_files(config):
    """查找所选小说的预处理 BIDS 事件文件。"""
    root = Path(config["root"])
    eeg_root = root / config["preprocessed_dir"]
    files = []
    for novel in select_novels(config.get("novel", "all")):
        files.extend(eeg_root.glob(f"sub-*/ses-{novel}/eeg/*_events.tsv"))
    if not files:
        raise FileNotFoundError(f"没有找到 ChineseEEG 事件文件：{eeg_root}")
    return sorted(files)


def parse_recording(events_path):
    """从文件名解析被试、小说和 run，不读取 EEG 数据。"""
    match = EVENT_FILE_PATTERN.match(events_path.name)
    if match is None:
        raise ValueError(f"无法解析事件文件名：{events_path.name}")

    subject_id = f"sub-{match.group('subject')}"
    novel = match.group("novel")
    run_id = int(match.group("run"))
    return subject_id, novel, run_id


def stimulus_path(config, novel, run_id):
    """返回一个 run 的分行文本文件。"""
    return (
        Path(config["root"])
        / config["stimulus_dir"]
        / novel
        / f"segmented_Chinense_novel_run_{run_id}.xlsx"
    )


def embedding_path(config, novel, run_id):
    """返回一个 run 的行级 BERT 向量文件。"""
    return (
        Path(config["root"])
        / config["text_embedding_dir"]
        / f"{novel}_text_embedding"
        / f"text_embedding_run_{run_id}.npy"
    )


def read_run_metadata(config, novel, run_id):
    """读取行文本并只检查 BERT 数组形状，不把向量写入事件表。"""
    text_path = stimulus_path(config, novel, run_id)
    bert_path = embedding_path(config, novel, run_id)
    if not text_path.exists():
        raise FileNotFoundError(f"缺少分行文本：{text_path}")
    if not bert_path.exists():
        raise FileNotFoundError(f"缺少 BERT 文件：{bert_path}")

    text_table = pd.read_excel(text_path)
    if "Chinese_text" not in text_table.columns:
        raise ValueError(f"分行文本缺少 Chinese_text 字段：{text_path}")
    if text_table["Chinese_text"].isna().any():
        raise ValueError(f"分行文本含空值：{text_path}")
    texts = [str(text).strip() for text in text_table["Chinese_text"]]

    embeddings = np.load(bert_path, mmap_mode="r")
    expected_dimension = int(config["embedding_dimension"])
    if embeddings.shape != (len(texts), expected_dimension):
        raise ValueError(
            f"BERT 形状与文本行不一致：{bert_path}，"
            f"实际 {embeddings.shape}，预期 {(len(texts), expected_dimension)}"
        )
    return texts, bert_path


def read_sampling_rate(events_path):
    """从 BrainVision 旁侧 JSON 读取采样率，不打开 EEG 信号文件。"""
    sidecar_path = events_path.with_name(
        events_path.name.replace("_events.tsv", "_eeg.json")
    )
    if not sidecar_path.exists():
        raise FileNotFoundError(f"缺少 EEG 旁侧文件：{sidecar_path}")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return float(metadata["SamplingFrequency"])


def align_row_markers(events, texts, events_path):
    """从首个章节标记开始，将每个文本行与一对 ROWS/ROWE 对齐。"""
    first_text = texts[0]
    if not first_text.isdigit():
        raise ValueError(f"文本首行不是章节编号：{events_path}")

    chapter_marker = f"CH{int(first_text):02d}"
    chapter_indices = events.index[events["trial_type"].eq(chapter_marker)]
    if chapter_indices.empty:
        raise ValueError(f"缺少章节标记 {chapter_marker}：{events_path}")

    formal_events = events.loc[chapter_indices[0] :]
    row_starts = formal_events[formal_events["trial_type"].eq("ROWS")].reset_index(drop=True)
    row_stops = formal_events[formal_events["trial_type"].eq("ROWE")].reset_index(drop=True)
    if len(row_starts) != len(texts) or len(row_stops) != len(texts):
        raise ValueError(
            f"ROWS/ROWE 与文本行数不一致：{events_path}，"
            f"文本 {len(texts)}，ROWS {len(row_starts)}，ROWE {len(row_stops)}"
        )
    if (row_starts["sample"].to_numpy() >= row_stops["sample"].to_numpy()).any():
        raise ValueError(f"存在 ROWS 不早于 ROWE 的行：{events_path}")
    return row_starts, row_stops


def highlighted_character_count(text):
    """按呈现程序规则计算真正高亮的字符数，忽略排版空白。"""
    return sum(
        character not in PRESENTATION_PUNCTUATION and not character.isspace()
        for character in text
    )


def overlaps_bad_segment(window_start, window_stop, bad_intervals):
    """判断左闭右开固定窗口是否与任一坏段相交。"""
    return any(
        window_start < bad_stop and window_stop > bad_start
        for bad_start, bad_stop in bad_intervals
    )


def build_bad_intervals(events, sampling_rate):
    """把 bad 事件转换为采样点区间。"""
    intervals = []
    for event in events[events["trial_type"].eq("bad")].itertuples():
        start = int(event.sample)
        stop = start + int(round(float(event.duration) * sampling_rate))
        intervals.append((start, stop))
    return intervals


def exclusion_reason(is_chapter, character_count, target_count, inside_row, overlaps_bad):
    """汇总一行不能用于训练的原因。"""
    reasons = []
    if is_chapter:
        reasons.append("章节编号")
    if character_count != target_count:
        reasons.append(f"高亮字符数不是{target_count}")
    if not inside_row:
        reasons.append("固定窗口越过ROWE")
    if overlaps_bad:
        reasons.append("固定窗口与bad区间重叠")
    return "；".join(reasons)


def build_recording_rows(events_path, config, texts, bert_path):
    """将一个 recording 转成一行文本对应一条记录的事件表。"""
    subject_id, novel, run_id = parse_recording(events_path)
    eeg_path = events_path.with_name(events_path.name.replace("_events.tsv", "_eeg.vhdr"))
    if not eeg_path.exists():
        raise FileNotFoundError(f"缺少对应 EEG 头文件：{eeg_path}")

    sampling_rate = read_sampling_rate(events_path)
    expected_rate = float(config["sampling_rate_hz"])
    if sampling_rate != expected_rate:
        raise ValueError(
            f"采样率不符合协议：{events_path}，实际 {sampling_rate}，预期 {expected_rate}"
        )

    events = pd.read_csv(events_path, sep="\t")
    row_starts, row_stops = align_row_markers(events, texts, events_path)
    bad_intervals = build_bad_intervals(events, sampling_rate)
    window_samples = int(round(float(config["window_seconds"]) * sampling_rate))
    target_count = int(config["highlighted_character_count"])
    recording_id = f"{subject_id}_{novel}_run-{run_id:02d}"
    rows = []

    for row_index, text in enumerate(texts):
        row_start = row_starts.iloc[row_index]
        row_stop = row_stops.iloc[row_index]
        start_sample = int(row_start["sample"])
        stop_sample = int(row_stop["sample"])
        window_stop = start_sample + window_samples
        character_count = highlighted_character_count(text)
        is_chapter = text.isdigit()
        inside_row = window_stop <= stop_sample
        overlaps_bad = overlaps_bad_segment(start_sample, window_stop, bad_intervals)
        reason = exclusion_reason(
            is_chapter, character_count, target_count, inside_row, overlaps_bad
        )
        rows.append(
            {
                "dataset": "ChineseEEG",
                "subject_id": subject_id,
                "novel": novel,
                "run_id": run_id,
                "recording_id": recording_id,
                "row_id": row_index + 1,
                "text": text,
                "text_id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "eeg_path": eeg_path.name,
                "row_start_sample": start_sample,
                "row_stop_sample": stop_sample,
                "window_start_sample": start_sample,
                "window_stop_sample": window_stop,
                "fixed_sample_count": window_samples,
                "row_start_seconds": float(row_start["onset"]),
                "row_stop_seconds": float(row_stop["onset"]),
                "measured_duration_seconds": float(row_stop["onset"] - row_start["onset"]),
                "highlighted_character_count": character_count,
                "bert_path": bert_path.name,
                "bert_row_index": row_index,
                "is_chapter_number": is_chapter,
                "overlaps_bad_segment": overlaps_bad,
                "window_inside_row": inside_row,
                "is_trainable": reason == "",
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_event_table(config):
    """构造全部行级记录；同一 run 的文本和 BERT 元数据只读取一次。"""
    metadata_cache = {}
    tables = []
    for events_path in find_event_files(config):
        _, novel, run_id = parse_recording(events_path)
        key = (novel, run_id)
        if key not in metadata_cache:
            metadata_cache[key] = read_run_metadata(config, novel, run_id)
        texts, bert_path = metadata_cache[key]
        tables.append(build_recording_rows(events_path, config, texts, bert_path))
    return pd.concat(tables, ignore_index=True)


def split_run_ids(run_ids, novel, config):
    """读取显式 run 划分，拒绝随机生成或遗漏 run。"""
    if novel not in config["by_novel"]:
        raise ValueError(f"尚未为 {novel} 固定 train/val/test run。")
    novel_split = config["by_novel"][novel]
    mapping = {
        run_id: split_name
        for split_name in ("train", "val", "test")
        for run_id in novel_split[split_name]
    }
    configured_runs = [
        run_id
        for split_name in ("train", "val", "test")
        for run_id in novel_split[split_name]
    ]
    if len(configured_runs) != len(set(configured_runs)):
        raise ValueError(f"{novel} 的同一 run 被分配到多个 split。")
    if set(run_ids) != set(configured_runs):
        raise ValueError(
            f"{novel} 的 run 配置与实际数据不一致："
            f"实际 {sorted(run_ids)}，配置 {sorted(configured_runs)}"
        )
    return mapping


def split_event_table(event_table, config):
    """按“小说 + run”划分，所有被试的同一 run 使用相同 split。"""
    result = event_table.copy()
    result["split"] = ""
    for novel in result["novel"].drop_duplicates():
        novel_rows = result["novel"].eq(novel)
        run_split = split_run_ids(
            result.loc[novel_rows, "run_id"].unique(), novel, config
        )
        result.loc[novel_rows, "split"] = result.loc[novel_rows, "run_id"].map(run_split)
    return result


def audit_run_split(event_table):
    """检查 run 泄露、空划分和固定窗口采样点数。"""
    run_table = event_table[["novel", "run_id", "split"]].drop_duplicates()
    leaked_runs = run_table.groupby(["novel", "run_id"])["split"].nunique()
    if (leaked_runs > 1).any():
        raise RuntimeError("发现同一 run 跨越多个 split。")
    if event_table["split"].eq("").any():
        raise RuntimeError("存在未分配 split 的行。")
    if event_table["fixed_sample_count"].nunique() != 1:
        raise RuntimeError("固定窗口采样点数不唯一。")
    return run_table.sort_values(["novel", "run_id"]).reset_index(drop=True)


def rename_fields(event_table, field_names=None):
    """按配置将行级事件表字段改为中文或其他名称。"""
    return event_table.rename(columns=field_names or {}).copy()


def parse_boolean(value):
    """把 Excel 中的布尔值安全转换为 bool。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "是"}:
        return True
    if text in {"false", "0", "no", "否"}:
        return False
    raise ValueError(f"无法解析布尔值：{value!r}")


def load_split_event_table(path, split):
    """读取可训练的行级记录；不会打开任何 EEG 文件。"""
    table = pd.read_excel(path)
    required = {
        "被试编号",
        "小说",
        "Run编号",
        "记录编号",
        "行编号",
        "文本ID",
        "EEG文件",
        "窗口开始采样点",
        "窗口结束采样点",
        "固定窗口采样点数",
        "BERT文件",
        "BERT行索引0基",
        "是否可训练",
        "数据划分",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"事件表缺少字段：{missing}")

    trainable = table["是否可训练"].map(parse_boolean)
    selected = table[trainable & table["数据划分"].eq(split)].copy()
    if selected.empty:
        raise ValueError(f"{split} 没有可训练行。")
    return selected.sort_values(["记录编号", "行编号"]).reset_index(drop=True)


def resolve_eeg_path(row, config):
    """由事件表中的文件名恢复 BrainVision 头文件路径。"""
    return (
        Path(config["root"])
        / config["preprocessed_dir"]
        / str(row["被试编号"])
        / f"ses-{row['小说']}"
        / "eeg"
        / str(row["EEG文件"])
    )


def resolve_electrode_files(row, config):
    """定位当前被试会话的电极坐标与坐标系文件。"""
    eeg_dir = resolve_eeg_path(row, config).parent
    prefix = f"{row['被试编号']}_ses-{row['小说']}_space-CapTrak"
    return (
        eeg_dir / f"{prefix}_electrodes.tsv",
        eeg_dir / f"{prefix}_coordsystem.json",
    )


@lru_cache(maxsize=64)
def read_channel_positions(electrodes_path, coordsystem_path, channel_names):
    """按 BrainVision 通道顺序读取 CapTrak 三维电极坐标。"""
    electrodes_path = Path(electrodes_path)
    coordsystem_path = Path(coordsystem_path)
    if not electrodes_path.exists() or not coordsystem_path.exists():
        raise FileNotFoundError(f"缺少电极坐标文件：{electrodes_path.parent}")
    coordinate_system = json.loads(coordsystem_path.read_text(encoding="utf-8"))
    if coordinate_system.get("EEGCoordinateUnits") != "m":
        raise ValueError(f"电极坐标单位不是米：{coordsystem_path}")

    table = pd.read_csv(electrodes_path, sep="\t")
    required = {"name", "x", "y", "z"}
    if not required.issubset(table.columns) or table["name"].duplicated().any():
        raise ValueError(f"电极坐标表字段无效：{electrodes_path}")
    table = table.set_index("name")
    channel_names = tuple(channel_names)
    if set(table.index) != set(channel_names):
        raise ValueError(f"电极坐标与 EEG 通道名称不一致：{electrodes_path}")
    positions = table.loc[list(channel_names), ["x", "y", "z"]].to_numpy(np.float32)
    if positions.shape != (len(channel_names), 3) or not np.isfinite(positions).all():
        raise ValueError(f"电极坐标包含缺失或非有限值：{electrodes_path}")
    positions.setflags(write=False)
    return positions


def resolve_embedding_path(row, config):
    """由事件表中的文件名恢复行级 BERT 文件路径。"""
    return (
        Path(config["root"])
        / config["text_embedding_dir"]
        / f"{row['小说']}_text_embedding"
        / str(row["BERT文件"])
    )


@lru_cache(maxsize=64)
def read_brainvision_header(path):
    """读取训练所需的 BrainVision 头信息。"""
    header_path = Path(path)
    sections = {}
    section = None
    for raw_line in header_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            sections[section] = {}
            continue
        if section and "=" in line:
            key, value = line.split("=", 1)
            sections[section][key] = value

    common = sections.get("Common Infos", {})
    binary = sections.get("Binary Infos", {})
    channel_info = sections.get("Channel Infos", {})
    channel_count = int(common["NumberOfChannels"])
    channel_names = []
    for index in range(1, channel_count + 1):
        value = channel_info[f"Ch{index}"]
        channel_names.append(value.split(",", 1)[0].replace(r"\1", ","))

    if common.get("DataOrientation") != "MULTIPLEXED":
        raise ValueError(f"只支持 MULTIPLEXED BrainVision 数据：{header_path}")
    if binary.get("BinaryFormat") != "IEEE_FLOAT_32":
        raise ValueError(f"只支持 IEEE_FLOAT_32 BrainVision 数据：{header_path}")
    return {
        "data_path": header_path.with_name(common["DataFile"]),
        "channel_names": tuple(channel_names),
        "channel_count": channel_count,
    }


@lru_cache(maxsize=64)
def open_eeg_array(path):
    """以内存映射方式打开连续 EEG，避免重复载入整段 recording。"""
    header = read_brainvision_header(path)
    data_path = header["data_path"]
    value_count = data_path.stat().st_size // np.dtype("<f4").itemsize
    if value_count % header["channel_count"] != 0:
        raise ValueError(f"EEG 文件大小与通道数不匹配：{data_path}")
    sample_count = value_count // header["channel_count"]
    return np.memmap(data_path, dtype="<f4", mode="r", shape=(sample_count, header["channel_count"]))


@lru_cache(maxsize=16)
def open_embedding_array(path):
    """以内存映射方式读取 BERT 向量。"""
    return np.load(path, mmap_mode="r")


def read_eeg_window(row, config, channel_names):
    """读取一个固定长度 EEG 窗口，输出形状为“通道 × 时间”。"""
    header_path = resolve_eeg_path(row, config)
    header = read_brainvision_header(str(header_path))
    if header["channel_names"] != tuple(channel_names):
        raise ValueError(f"EEG 通道顺序不一致：{header_path}")

    start = int(row["窗口开始采样点"])
    stop = int(row["窗口结束采样点"])
    expected_samples = int(row["固定窗口采样点数"])
    if stop - start != expected_samples:
        raise ValueError(f"事件表窗口不是固定长度：{header_path}")

    continuous = open_eeg_array(str(header_path))
    if start < 0 or stop > len(continuous):
        raise IndexError(f"EEG 窗口越界：{header_path} [{start}, {stop})")
    return np.asarray(continuous[start:stop].T, dtype=np.float32).copy()


def text_key(text_id):
    """将文本哈希稳定映射为 PyTorch 可用的正整数。"""
    return np.int64(int(str(text_id)[:15], 16))


class ChineseEEGRowDataset:
    """按需读取固定窗口 EEG 与对应 BERT 目标。"""

    def __init__(
        self,
        table,
        config,
        source_indices=None,
        zero_eeg=False,
        subject_to_index=None,
    ):
        self.table = table.reset_index(drop=True)
        self.config = config
        self.source_indices = (
            np.arange(len(self.table), dtype=np.int64)
            if source_indices is None
            else np.asarray(source_indices, dtype=np.int64)
        )
        if len(self.source_indices) != len(self.table):
            raise ValueError("source_indices 长度与事件表不一致。")
        self.zero_eeg = bool(zero_eeg)
        first_source = self.table.iloc[int(self.source_indices[0])]
        first_header = read_brainvision_header(str(resolve_eeg_path(first_source, config)))
        self.channel_names = first_header["channel_names"]
        electrode_files = resolve_electrode_files(first_source, config)
        self.channel_positions = read_channel_positions(
            str(electrode_files[0]), str(electrode_files[1]), self.channel_names
        )
        self.window_samples = int(self.table.iloc[0]["固定窗口采样点数"])
        subjects = sorted(str(value) for value in self.table["被试编号"].unique())
        self.subject_to_index = (
            {subject: index for index, subject in enumerate(subjects)}
            if subject_to_index is None
            else {str(subject): int(index) for subject, index in subject_to_index.items()}
        )
        unknown_subjects = sorted(set(subjects) - set(self.subject_to_index))
        if unknown_subjects:
            raise ValueError(f"受试者索引缺少：{unknown_subjects}")
        self.subject_count = len(self.subject_to_index)
        self.channel_mean = None
        self.channel_std = None

    def __len__(self):
        return len(self.table)

    def set_normalization(self, channel_mean, channel_std):
        """设置仅由训练集估计的逐通道标准化参数。"""
        mean = np.asarray(channel_mean, dtype=np.float32).reshape(-1, 1)
        std = np.asarray(channel_std, dtype=np.float32).reshape(-1, 1)
        if len(mean) != len(self.channel_names) or (std <= 0).any():
            raise ValueError("通道标准化参数无效。")
        self.channel_mean = mean
        self.channel_std = std

    def read_eeg(self, index, normalize=True):
        """读取指定表行的 EEG，可供训练集统计复用。"""
        row = self.table.iloc[int(index)]
        eeg = read_eeg_window(row, self.config, self.channel_names)
        if normalize:
            if self.channel_mean is None or self.channel_std is None:
                raise RuntimeError("尚未设置训练集标准化参数。")
            eeg = (eeg - self.channel_mean) / self.channel_std
        return eeg.astype(np.float32, copy=False)

    def __getitem__(self, index):
        target_row = self.table.iloc[int(index)]
        if self.zero_eeg:
            eeg = np.zeros(
                (len(self.channel_names), self.window_samples), dtype=np.float32
            )
        else:
            source_index = int(self.source_indices[int(index)])
            eeg = self.read_eeg(source_index, normalize=True)

        bert_path = resolve_embedding_path(target_row, self.config)
        embeddings = open_embedding_array(str(bert_path))
        bert_index = int(target_row["BERT行索引0基"])
        target = np.asarray(embeddings[bert_index], dtype=np.float32).copy()
        identifier = str(target_row["文本ID"])
        subject_id = str(target_row["被试编号"])
        return {
            "eeg": np.ascontiguousarray(eeg),
            "text_embedding": target,
            "text_key": text_key(identifier),
            "text_id": identifier,
            "subject_id": subject_id,
            "subject_index": np.int64(self.subject_to_index[subject_id]),
            "recording_id": str(target_row["记录编号"]),
            "run_id": np.int64(target_row["Run编号"]),
            "row_id": np.int64(target_row["行编号"]),
        }


def build_wrong_row_indices(table):
    """为每条验证记录选择同一 recording 内文本不同的 EEG 行。"""
    table = table.reset_index(drop=True)
    source_indices = np.empty(len(table), dtype=np.int64)
    for _, group in table.groupby("记录编号", sort=False):
        positions = group.index.to_numpy(dtype=np.int64)
        identifiers = table.loc[positions, "文本ID"].astype(str).to_numpy()
        if len(positions) < 2:
            raise ValueError("recording 内不足两行，无法构造错行对照。")
        for local_index, position in enumerate(positions):
            for offset in range(1, len(positions)):
                candidate = (local_index + offset) % len(positions)
                if identifiers[candidate] != identifiers[local_index]:
                    source_indices[position] = positions[candidate]
                    break
            else:
                raise ValueError("recording 内没有文本不同的错行候选。")

    if not table.loc[source_indices, "记录编号"].reset_index(drop=True).eq(
        table["记录编号"]
    ).all():
        raise RuntimeError("错行对照跨越了 recording。")
    if table.loc[source_indices, "文本ID"].reset_index(drop=True).eq(
        table["文本ID"]
    ).any():
        raise RuntimeError("错行对照仍含相同文本。")
    return source_indices


def load_eeg(path, preload=False):
    """训练取样时才显式读取 BrainVision EEG。"""
    import mne

    return mne.io.read_raw_brainvision(path, preload=preload)
