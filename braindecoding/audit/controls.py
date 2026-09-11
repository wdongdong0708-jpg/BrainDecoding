"""验证集 temporal shift 与 donor swap 的确定性事件映射。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DONOR_SEEDS = tuple(range(20))


def json_bytes(value) -> bytes:
    """使用固定格式生成 UTF-8 JSON 字节。"""
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def payload_sha256(payload: dict, field: str) -> str:
    """计算排除自指摘要字段后的规范 JSON SHA-256。"""
    content = dict(payload)
    content.pop(field, None)
    canonical = json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def with_payload_sha256(payload: dict, field: str) -> dict:
    """返回带自校验摘要的副本。"""
    result = dict(payload)
    result[field] = payload_sha256(result, field)
    return result


def validate_payload_sha256(payload: dict, field: str) -> None:
    """拒绝摘要缺失或内容被修改的协议资产。"""
    expected = str(payload.get(field, ""))
    actual = payload_sha256(payload, field)
    if not expected or expected != actual:
        raise ValueError(
            f"协议资产 SHA-256 不一致：期望 {expected or '<missing>'}，实际 {actual}"
        )


def file_sha256(path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    normalized = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"无法解析是否可训练字段：{unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def _require_columns(table: pd.DataFrame, columns) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"事件表缺少生成对照映射所需字段：{missing}")


def require_validation_split(split: str) -> None:
    """本阶段只允许 validation，且不提供 test 绕过开关。"""
    if str(split) != "val":
        raise ValueError("正式神经对照当前只允许 split='val'；禁止访问 test 脑数据。")


def prepare_validation_events(table: pd.DataFrame, split: str = "val") -> pd.DataFrame:
    """筛选可评价 validation 事件，并保留原记录内顺序。"""
    require_validation_split(split)
    _require_columns(
        table,
        (
            "event_id",
            "subject_id",
            "recording_id",
            "normalized_word",
            "sentence_uid",
            "split",
            "is_trainable",
            "window_start_seconds",
            "window_stop_seconds",
        ),
    )
    result = table.copy()
    if "记录内序号" in result:
        record_order = pd.to_numeric(result["记录内序号"], errors="coerce")
        if record_order.isna().any():
            raise ValueError("记录内序号必须是有效整数。")
        result["_control_record_order"] = record_order.astype(int)
    else:
        # 历史缓存没有中文列时，沿用已经锁定的当前行序。
        result["_control_record_order"] = result.groupby(
            "recording_id", sort=False, dropna=False
        ).cumcount()

    mask = result["split"].eq("val") & _parse_bool_series(result["is_trainable"])
    if "window_complete" in result:
        mask &= _parse_bool_series(result["window_complete"])
    result = result.loc[mask].copy()
    if result.empty:
        raise ValueError("validation 中没有具备完整输入窗口的可评价事件。")
    if result["event_id"].astype(str).duplicated().any():
        raise ValueError("validation 事件编号必须唯一。")
    if not result["split"].eq("val").all():
        raise AssertionError("控制映射意外包含非 validation 事件。")

    result["event_id"] = result["event_id"].astype(str)
    result["subject_id"] = result["subject_id"].astype(str)
    result["recording_id"] = result["recording_id"].astype(str)
    result["normalized_word"] = result["normalized_word"].astype(str)
    for column in ("window_start_seconds", "window_stop_seconds"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or not np.isfinite(result[column]).all():
            raise ValueError(f"{column} 必须是有限秒数。")
    if result["window_stop_seconds"].le(result["window_start_seconds"]).any():
        raise ValueError("脑输入窗口必须具有正时长。")

    return result.sort_values(
        ["subject_id", "recording_id", "_control_record_order", "event_id"],
        kind="stable",
    ).reset_index(drop=True)


def windows_do_not_overlap(left, right) -> bool:
    """两个半开时间窗相接可以，但不得具有正长度重叠。"""
    return bool(
        float(left["window_stop_seconds"])
        <= float(right["window_start_seconds"])
        or float(right["window_stop_seconds"])
        <= float(left["window_start_seconds"])
    )


def _mapping_row(target, source, *, include_words: bool) -> dict:
    row = {
        "recording": str(target["recording_id"]),
        "source_event_id": str(source["event_id"]),
        "source_record_order": int(source["_control_record_order"]),
        "source_window_start": float(source["window_start_seconds"]),
        "source_window_stop": float(source["window_stop_seconds"]),
        "subject": str(target["subject_id"]),
        "target_event_id": str(target["event_id"]),
        "target_record_order": int(target["_control_record_order"]),
        "target_window_start": float(target["window_start_seconds"]),
        "target_window_stop": float(target["window_stop_seconds"]),
    }
    if include_words:
        row["source_standard_word"] = str(source["normalized_word"])
        row["target_standard_word"] = str(target["normalized_word"])
    return row


def build_identity_mapping(
    events: pd.DataFrame, *, dataset: str, event_table_sha256: str
) -> dict:
    """为 clean 条件生成逐事件 identity source mapping。"""
    rows = [
        _mapping_row(row, row, include_words=False)
        for _, row in events.iterrows()
    ]
    return with_payload_sha256(
        {
            "algorithm": "identity",
            "control_condition": "clean",
            "dataset": str(dataset),
            "eligible_count": len(rows),
            "event_table_sha256": str(event_table_sha256),
            "mappings": rows,
            "split": "val",
            "test_neural_data_opened": False,
        },
        "mapping_sha256",
    )


def _temporal_offsets(event_count: int) -> list[int]:
    midpoint = int(event_count) // 2
    return sorted(
        range(1, int(event_count)),
        key=lambda offset: (abs(offset - midpoint), offset),
    )


def build_temporal_shift_mapping(
    events: pd.DataFrame, *, dataset: str, event_table_sha256: str
) -> dict:
    """在同受试者、同记录内生成最接近半记录距离的确定性循环错位。"""
    rows = []
    ineligible = []
    for _, group in events.groupby(
        ["subject_id", "recording_id"], sort=False, dropna=False
    ):
        group = group.reset_index(drop=True)
        offsets = _temporal_offsets(len(group))
        for target_index, target in group.iterrows():
            source = None
            for offset in offsets:
                candidate = group.iloc[(target_index + offset) % len(group)]
                if candidate["event_id"] == target["event_id"]:
                    continue
                if windows_do_not_overlap(target, candidate):
                    source = candidate
                    break
            if source is None:
                ineligible.append(str(target["event_id"]))
            else:
                rows.append(_mapping_row(target, source, include_words=False))
    return with_payload_sha256(
        {
            "algorithm": {
                "candidate": "(i + offset) mod n",
                "offset_order": "abs(offset-floor(n/2)) then smaller offset",
                "uses_word_labels": False,
            },
            "control_condition": "temporal_shift",
            "dataset": str(dataset),
            "eligible_count": len(rows),
            "event_table_sha256": str(event_table_sha256),
            "ineligible_count": len(ineligible),
            "ineligible_event_ids": ineligible,
            "mappings": rows,
            "split": "val",
            "test_neural_data_opened": False,
        },
        "mapping_sha256",
    )


def donor_candidates(events: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """建立每个 target 的全部同记录、异词、非重叠 donor 候选。"""
    candidates = {}
    for _, group in events.groupby(
        ["subject_id", "recording_id"], sort=False, dropna=False
    ):
        group = group.reset_index(drop=True)
        starts = group["window_start_seconds"].to_numpy(dtype=float)
        stops = group["window_stop_seconds"].to_numpy(dtype=float)
        durations = stops - starts
        words = group["normalized_word"].astype(str).to_numpy()
        event_ids = group["event_id"].astype(str).to_numpy()
        for index, event_id in enumerate(event_ids):
            legal = (
                ((stops <= starts[index]) | (starts >= stops[index]))
                & np.isclose(durations, durations[index], rtol=0.0, atol=1e-9)
                & (words != words[index])
                & (event_ids != event_id)
            )
            candidates[event_id] = tuple(event_ids[np.flatnonzero(legal)])
    return candidates


def build_donor_swap_mapping(
    events: pd.DataFrame,
    candidates: dict[str, tuple[str, ...]],
    *,
    dataset: str,
    event_table_sha256: str,
    seed: int,
) -> dict:
    """用 default_rng(seed) 为每个 target 独立均匀抽取一个合法 donor。"""
    seed = int(seed)
    random = np.random.default_rng(seed)
    by_event = events.set_index("event_id", drop=False)
    rows = []
    ineligible = []
    for event_id in events["event_id"].astype(str):
        legal = candidates[event_id]
        if not legal:
            ineligible.append(event_id)
            continue
        source_id = str(random.choice(np.asarray(legal, dtype=object)))
        rows.append(
            _mapping_row(
                by_event.loc[event_id],
                by_event.loc[source_id],
                include_words=True,
            )
        )
    return with_payload_sha256(
        {
            "algorithm": {
                "candidate_sampling": "uniform_per_target",
                "random_generator": "numpy.random.default_rng",
                "replacement_across_targets": True,
            },
            "control_condition": "donor_swap",
            "dataset": str(dataset),
            "eligible_count": len(rows),
            "event_table_sha256": str(event_table_sha256),
            "ineligible_count": len(ineligible),
            "ineligible_event_ids": ineligible,
            "mappings": rows,
            "seed": seed,
            "split": "val",
            "test_neural_data_opened": False,
        },
        "mapping_sha256",
    )


def build_core_audit_queries(
    events: pd.DataFrame,
    temporal_mapping: dict,
    donor_mappings: list[dict],
    *,
    dataset: str,
    event_table_sha256: str,
) -> dict:
    """冻结 clean、temporal、全部 donor seed 共同可评价的查询。"""
    temporal = {
        row["target_event_id"] for row in temporal_mapping["mappings"]
    }
    donor_sets = [
        {row["target_event_id"] for row in mapping["mappings"]}
        for mapping in donor_mappings
    ]
    donor = set.intersection(*donor_sets) if donor_sets else set()
    clean = set(events["event_id"].astype(str))
    common = clean & temporal & donor
    ordered = [
        event_id
        for event_id in events["event_id"].astype(str)
        if event_id in common
    ]
    return with_payload_sha256(
        {
            "clean_evaluable_count": len(clean),
            "dataset": str(dataset),
            "donor_eligible_count": len(donor),
            "event_ids": ordered,
            "event_table_sha256": str(event_table_sha256),
            "query_count": len(ordered),
            "query_rule": "clean_evaluable intersect temporal_eligible intersect donor_eligible",
            "split": "val",
            "temporal_eligible_count": len(temporal),
            "test_neural_data_opened": False,
        },
        "query_manifest_sha256",
    )


def build_control_assets(
    table: pd.DataFrame,
    *,
    dataset: str,
    event_table_sha256: str,
    split: str = "val",
    seeds=DONOR_SEEDS,
    status: str = "validation_only",
) -> dict[str, dict]:
    """一次生成同一数据集全部公共 validation 映射资产。"""
    events = prepare_validation_events(table, split=split)
    clean = build_identity_mapping(
        events, dataset=dataset, event_table_sha256=event_table_sha256
    )
    temporal = build_temporal_shift_mapping(
        events, dataset=dataset, event_table_sha256=event_table_sha256
    )
    candidates = donor_candidates(events)
    donor_mappings = [
        build_donor_swap_mapping(
            events,
            candidates,
            dataset=dataset,
            event_table_sha256=event_table_sha256,
            seed=int(seed),
        )
        for seed in seeds
    ]
    core = build_core_audit_queries(
        events,
        temporal,
        donor_mappings,
        dataset=dataset,
        event_table_sha256=event_table_sha256,
    )
    assets = {
        "clean_identity_mapping.json": clean,
        "core_audit_queries.json": core,
        "temporal_shift_mapping.json": temporal,
    }
    for mapping in donor_mappings:
        assets[f"donor_swap_seed{int(mapping['seed']):02d}.json"] = mapping
    for name, payload in list(assets.items()):
        sha_field = (
            "query_manifest_sha256"
            if name == "core_audit_queries.json"
            else "mapping_sha256"
        )
        updated = dict(payload)
        updated["status"] = str(status)
        assets[name] = with_payload_sha256(updated, sha_field)
    return assets


def write_control_assets(assets: dict[str, dict], output_dir) -> list[Path]:
    """稳定写入映射资产，不包含运行时间等易变字段。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, payload in sorted(assets.items()):
        path = output_dir / name
        path.write_bytes(json_bytes(payload))
        paths.append(path)
    return paths


def load_control_asset(path, sha_field: str) -> dict:
    """读取并验证映射或查询资产。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_payload_sha256(payload, sha_field)
    if payload.get("split") != "val":
        raise ValueError("控制资产不是 validation-only。")
    return payload
