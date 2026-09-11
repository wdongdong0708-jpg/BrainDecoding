"""从已冻结事件表生成论文协议资产，不读取脑信号。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "manifests"
DEFAULT_CHINESE_EVENT_TABLE = (
    PROJECT_ROOT
    / "tasks"
    / "word_decoding"
    / "ChineseEEG2_LittlePrince"
    / "cache"
    / "ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_semantic_v1.csv"
)
DEFAULT_SMN_EVENT_TABLE = (
    PROJECT_ROOT
    / "tasks"
    / "word_decoding"
    / "SMN4Lang"
    / "cache"
    / "SMN4Lang_events_all_words_sentence_groups.csv"
)
PRIMARY_VOCABULARY_SIZES = (20, 50, 100, 150)
CHINESE_SUBJECTS = tuple(f"sub-{index:02d}" for index in range(1, 9))
SMN_EXPECTED_SUBJECTS = tuple(f"sub-{index:02d}" for index in range(1, 13))


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    """使用稳定键序和 UTF-8 生成可重复写入的 JSON。"""
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def manifest_sha256(manifest: dict) -> str:
    """计算去掉自指字段后的规范 JSON 摘要。"""
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def with_manifest_sha256(manifest: dict) -> dict:
    """返回带稳定自校验摘要的 manifest 副本。"""
    result = dict(manifest)
    result["manifest_sha256"] = manifest_sha256(result)
    return result


def validate_manifest_sha256(manifest: dict) -> None:
    """验证 manifest 自校验摘要。"""
    expected = str(manifest.get("manifest_sha256", ""))
    actual = manifest_sha256(manifest)
    if not expected or expected != actual:
        raise ValueError(
            f"manifest SHA-256 不一致：期望 {expected or '<missing>'}，实际 {actual}"
        )


def _clean_word(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _bool_series(values: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"无法解析是否可训练字段：{unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def _require_columns(table: pd.DataFrame, columns) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"事件表缺少协议资产所需字段：{missing}")


def validate_split_units(table: pd.DataFrame) -> None:
    """验证一个材料划分单元不会跨越 train/val/test。"""
    _require_columns(table, ("_split_unit", "split"))
    allowed = {"train", "val", "test"}
    observed = set(table["split"].astype(str))
    if not observed.issubset(allowed):
        raise ValueError(f"事件表含未知数据划分：{sorted(observed - allowed)}")
    split_counts = table.groupby("_split_unit", sort=False)["split"].nunique()
    crossed = sorted(split_counts[split_counts.gt(1)].index.astype(str))
    if crossed:
        raise ValueError(f"划分单元跨 split：{crossed[:10]}")


def canonical_words(
    table: pd.DataFrame,
    *,
    deduplicate_by,
    order_by,
    split: str | None = None,
    trainable_only: bool = False,
) -> pd.DataFrame:
    """按数据集合同去除受试者重复，保留一份材料词序列。"""
    required = {
        "normalized_word",
        *deduplicate_by,
        *order_by,
    }
    if split is not None:
        required.add("split")
    if trainable_only:
        required.add("is_trainable")
    _require_columns(table, required)

    selected = table
    if split is not None:
        selected = selected[selected["split"].eq(split)]
    if trainable_only:
        selected = selected[_bool_series(selected["is_trainable"])]
    selected = selected.copy()
    selected["_protocol_word"] = selected["normalized_word"].map(_clean_word)

    conflicts = selected.groupby(list(deduplicate_by), dropna=False)[
        "_protocol_word"
    ].nunique(dropna=False)
    conflicts = conflicts[conflicts.gt(1)]
    if not conflicts.empty:
        raise ValueError(
            f"同一材料词位置对应多个标准词：{list(conflicts.index[:5])}"
        )

    return (
        selected.sort_values(list(order_by), kind="stable")
        .drop_duplicates(list(deduplicate_by), keep="first")
        .reset_index(drop=True)
    )


def ranked_word_counts(words) -> tuple[list[str], dict[str, int]]:
    """按频次降序、词典序升序返回全部非空标准词。"""
    counts = Counter(_clean_word(word) for word in words)
    counts.pop("", None)
    ranked = sorted(counts, key=lambda word: (-int(counts[word]), word))
    return ranked, {word: int(counts[word]) for word in ranked}


def build_vocabulary_manifest(
    *,
    dataset: str,
    size: int,
    canonical_train_words: pd.DataFrame,
    source_units,
    event_table_path: Path,
    event_table_sha256: str,
    generator_sha256: str,
    status: str,
    normalization: str,
    scope: str,
) -> dict:
    """只从显式传入的训练材料生成候选词表 manifest。"""
    if not canonical_train_words["split"].eq("train").all():
        raise ValueError("候选词表输入包含非 train 事件。")
    ranked, all_counts = ranked_word_counts(canonical_train_words["_protocol_word"])
    vocabulary = ranked[: int(size)]
    if len(vocabulary) != int(size):
        raise ValueError(
            f"{dataset} 训练材料只有 {len(vocabulary)} 个非空标准词，无法生成 N={size}。"
        )
    manifest = {
        "asset_type": "candidate_vocabulary",
        "created_from_test": False,
        "dataset": dataset,
        "evaluation_support_used_for_selection": False,
        "event_table": _project_relative(event_table_path),
        "event_table_sha256": event_table_sha256,
        "generator": "experiments/generate_manifests.py",
        "generator_sha256": generator_sha256,
        "hash_contract": "sha256(canonical_json_without_manifest_sha256)",
        "normalization": normalization,
        "policy": "frequency_desc_lexical_tiebreak",
        "scope": scope,
        "source_split": "train",
        "source_units": list(source_units),
        "status": status,
        "vocabulary": vocabulary,
        "vocabulary_size": int(size),
        "word_counts": {word: all_counts[word] for word in vocabulary},
    }
    return with_manifest_sha256(manifest)


def build_story_reference(canonical_story_words: pd.DataFrame) -> dict[str, int]:
    """统计调用方明确提供的完整材料标准词，不读取候选词表。"""
    _, counts = ranked_word_counts(canonical_story_words["_protocol_word"])
    return counts


def build_ovmi_support_manifest(
    *,
    dataset: str,
    table: pd.DataFrame,
    vocabularies: dict[int, dict],
    generator_sha256: str,
    status: str,
    test_status: str,
) -> dict:
    """独立记录评价支持；结果不得用于改变冻结候选词。"""
    trainable = table[_bool_series(table["is_trainable"])].copy()
    support = {}
    for size, vocabulary_manifest in vocabularies.items():
        vocabulary = list(vocabulary_manifest["vocabulary"])
        split_support = {}
        for split in ("val", "test"):
            observed = {
                _clean_word(value)
                for value in trainable.loc[
                    trainable["split"].eq(split), "normalized_word"
                ]
            }
            observed.discard("")
            missing = [word for word in vocabulary if word not in observed]
            split_support[split] = {
                "full_ovmi_status": (
                    "available" if not missing else "unavailable_missing_true_samples"
                ),
                "missing_word_count": len(missing),
                "missing_words": missing,
                "supported_word_count": len(vocabulary) - len(missing),
            }
        support[f"N{size}"] = {
            "candidate_manifest_sha256": vocabulary_manifest["manifest_sha256"],
            "splits": split_support,
        }
    return with_manifest_sha256(
        {
            "asset_type": "ovmi_evaluation_support",
            "dataset": dataset,
            "full_ovmi_rule": {
                "missing_true_sample_action": "unavailable",
                "remove_missing_words": False,
                "report_coverage_and_retrieval": True,
                "silent_metric_substitution": False,
                "smooth_zero_rows": False,
            },
            "generator": "experiments/generate_manifests.py",
            "generator_sha256": generator_sha256,
            "status": status,
            "support_audit_used_for_vocabulary_selection": False,
            "test_status": test_status,
            "vocabularies": support,
        }
    )


def _project_relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _split_unit_lists(table: pd.DataFrame, *, trainable_only=False) -> dict:
    selected = table
    if trainable_only:
        selected = selected[_bool_series(selected["is_trainable"])]
    return {
        split: sorted(
            selected.loc[selected["split"].eq(split), "_split_unit"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        for split in ("train", "val", "test")
    }


def _prepare_chinese(table: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        table,
        (
            "chapter",
            "voice_version",
            "source_word_id",
            "chapter_word_index",
            "subject_id",
            "split",
            "is_trainable",
            "normalized_word",
        ),
    )
    result = table.copy()
    result["_split_unit"] = result["chapter"].map(
        lambda chapter: f"ChineseEEG2|littleprince|chapter-{int(chapter):02d}"
    )
    result["_material_unit"] = [
        f"ChineseEEG2|littleprince|chapter-{int(chapter):02d}|voice-{voice}"
        for chapter, voice in zip(result["chapter"], result["voice_version"])
    ]
    validate_split_units(result)
    return result


def _prepare_smn(table: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        table,
        (
            "run",
            "word_index",
            "subject_id",
            "split",
            "is_trainable",
            "normalized_word",
        ),
    )
    result = table.copy()
    result["_split_unit"] = result["run"].map(
        lambda run: f"SMN4Lang|story-{int(run):02d}"
    )
    validate_split_units(result)
    return result


def _story_assets(
    *,
    dataset: str,
    canonical_story: pd.DataFrame,
    source_units,
    deduplication_key,
    event_table_path: Path,
    event_table_sha256: str,
    generator_sha256: str,
    status: str,
) -> tuple[dict, dict]:
    reference = build_story_reference(canonical_story)
    reference_digest = hashlib.sha256(json_bytes(reference)).hexdigest()
    provenance = with_manifest_sha256(
        {
            "asset_type": "story_reference_provenance",
            "dataset": dataset,
            "deduplication_key": list(deduplication_key),
            "domain_reference": {"status": "not_frozen"},
            "event_or_text_source": _project_relative(event_table_path),
            "event_table_sha256": event_table_sha256,
            "generator": "experiments/generate_manifests.py",
            "generator_sha256": generator_sha256,
            "includes_all_target_splits": True,
            "includes_neurally_ineligible_story_words": True,
            "normalization": "event_table.normalized_word",
            "reference_sha256": reference_digest,
            "source_units": list(source_units),
            "status": status,
            "subject_repetitions_counted": False,
            "token_count": int(sum(reference.values())),
            "type_count": int(len(reference)),
        }
    )
    return reference, provenance


def _chinese_documents(
    event_table_path: Path, generator_sha256: str
) -> dict[str, dict]:
    table = _prepare_chinese(pd.read_csv(event_table_path, low_memory=False))
    subjects = tuple(sorted(table["subject_id"].astype(str).unique()))
    if subjects != CHINESE_SUBJECTS:
        raise ValueError(
            f"ChineseEEG2 正式受试者顺序不一致：期望 {CHINESE_SUBJECTS}，实际 {subjects}"
        )
    event_digest = file_sha256(event_table_path)
    canonical_train = canonical_words(
        table,
        deduplicate_by=("voice_version", "source_word_id"),
        order_by=(
            "voice_version",
            "chapter",
            "chapter_word_index",
            "subject_id",
        ),
        split="train",
        trainable_only=True,
    )
    source_units = sorted(canonical_train["_split_unit"].drop_duplicates())
    vocabularies = {
        size: build_vocabulary_manifest(
            dataset="ChineseEEG2",
            size=size,
            canonical_train_words=canonical_train,
            source_units=source_units,
            event_table_path=event_table_path,
            event_table_sha256=event_digest,
            generator_sha256=generator_sha256,
            status="frozen",
            normalization="event_table.normalized_word",
            scope="formal_eight_subject_passive_listening",
        )
        for size in PRIMARY_VOCABULARY_SIZES
    }
    canonical_story = canonical_words(
        table,
        deduplicate_by=("voice_version", "source_word_id"),
        order_by=(
            "voice_version",
            "chapter",
            "chapter_word_index",
            "subject_id",
        ),
    )
    story_units = sorted(canonical_story["_material_unit"].drop_duplicates())
    reference, provenance = _story_assets(
        dataset="ChineseEEG2",
        canonical_story=canonical_story,
        source_units=story_units,
        deduplication_key=("voice_version", "source_word_id"),
        event_table_path=event_table_path,
        event_table_sha256=event_digest,
        generator_sha256=generator_sha256,
        status="frozen",
    )

    unit_counts = table.groupby("_split_unit").agg(
        row_count=("_split_unit", "size"),
        trainable_event_count=("is_trainable", lambda values: int(_bool_series(values).sum())),
    )
    exclusions = {}
    for chapter in (14, 27):
        unit = f"ChineseEEG2|littleprince|chapter-{chapter:02d}"
        exclusions[unit] = {
            "configured_split": str(table.loc[table["_split_unit"].eq(unit), "split"].iloc[0]),
            "reason": "sub-05_recording_truncated_shared_eight_subject_exclusion",
            "row_count": int(unit_counts.loc[unit, "row_count"]),
            "trainable_event_count": int(unit_counts.loc[unit, "trainable_event_count"]),
        }
    split_manifest = with_manifest_sha256(
        {
            "asset_type": "split_and_subject_manifest",
            "confirmatory_status": "exploratory_or_replication_only",
            "context_contract": {
                "neural_context": "bounded_semantic_v1",
                "row": "historical_structure_prior_control",
                "word": {"model.use_transformer": False},
            },
            "dataset": "ChineseEEG2",
            "event_table": _project_relative(event_table_path),
            "event_table_sha256": event_digest,
            "generator": "experiments/generate_manifests.py",
            "generator_sha256": generator_sha256,
            "shared_excluded_units": exclusions,
            "split_units": _split_unit_lists(table),
            "status": "frozen",
            "subject_order": list(subjects),
            "subjects": list(subjects),
            "test_status": "previously_used_for_exploratory_analysis",
            "trainable_split_units": _split_unit_lists(table, trainable_only=True),
        }
    )
    support = build_ovmi_support_manifest(
        dataset="ChineseEEG2",
        table=table,
        vocabularies=vocabularies,
        generator_sha256=generator_sha256,
        status="frozen_support_audit",
        test_status="previously_used_for_exploratory_analysis",
    )
    documents = {
        "chineseeeg2/split_manifest.json": split_manifest,
        "chineseeeg2/story_reference.json": reference,
        "chineseeeg2/story_reference.provenance.json": provenance,
        "chineseeeg2/ovmi_support.json": support,
    }
    for size, manifest in vocabularies.items():
        documents[f"chineseeeg2/vocabulary_N{size}.json"] = manifest
    return documents


def _smn_documents(event_table_path: Path, generator_sha256: str) -> dict[str, dict]:
    table = _prepare_smn(pd.read_csv(event_table_path, low_memory=False))
    subjects = tuple(sorted(table["subject_id"].astype(str).unique()))
    if subjects != ("sub-01",):
        raise ValueError(
            "SMN4Lang development 资产只允许从当前 sub-01 缓存生成；"
            f"实际受试者为 {subjects}。"
        )
    event_digest = file_sha256(event_table_path)
    canonical_train = canonical_words(
        table,
        deduplicate_by=("run", "word_index"),
        order_by=("run", "word_index", "subject_id"),
        split="train",
        trainable_only=False,
    )
    source_units = sorted(canonical_train["_split_unit"].drop_duplicates())
    vocabularies = {
        size: build_vocabulary_manifest(
            dataset="SMN4Lang",
            size=size,
            canonical_train_words=canonical_train,
            source_units=source_units,
            event_table_path=event_table_path,
            event_table_sha256=event_digest,
            generator_sha256=generator_sha256,
            status="development_only",
            normalization="event_table.normalized_word",
            scope="sub01_labels_awaiting_multisubject_data",
        )
        for size in PRIMARY_VOCABULARY_SIZES
    }
    canonical_story = canonical_words(
        table,
        deduplicate_by=("run", "word_index"),
        order_by=("run", "word_index", "subject_id"),
    )
    story_units = sorted(canonical_story["_split_unit"].drop_duplicates())
    reference, provenance = _story_assets(
        dataset="SMN4Lang",
        canonical_story=canonical_story,
        source_units=story_units,
        deduplication_key=("run", "word_index"),
        event_table_path=event_table_path,
        event_table_sha256=event_digest,
        generator_sha256=generator_sha256,
        status="frozen_from_dataset_annotations",
    )
    development_manifest = with_manifest_sha256(
        {
            "asset_type": "split_and_subject_manifest",
            "confirmatory_status": "eligible_after_multisubject_protocol_freeze",
            "context_contract": {
                "neural_context": "script_sentence_then_contiguous_chunks",
                "word": {"model.use_transformer": False},
            },
            "dataset": "SMN4Lang",
            "event_table": _project_relative(event_table_path),
            "event_table_sha256": event_digest,
            "expected_formal_subject_order": list(SMN_EXPECTED_SUBJECTS),
            "formal_manifest_status": "awaiting_multisubject_data",
            "generator": "experiments/generate_manifests.py",
            "generator_sha256": generator_sha256,
            "observed_subject_order": list(subjects),
            "split_units": _split_unit_lists(table),
            "status": "development_sub01",
            "subject_order": list(subjects),
            "test_label_status": "inspected_for_support_audit",
            "test_meg_status": "unopened",
            "trainable_split_units": _split_unit_lists(table, trainable_only=True),
        }
    )
    support = build_ovmi_support_manifest(
        dataset="SMN4Lang",
        table=table,
        vocabularies=vocabularies,
        generator_sha256=generator_sha256,
        status="development_sub01_support_audit",
        test_status="test_meg_unopened_labels_inspected",
    )
    documents = {
        "smn4lang/development_manifest_sub01.json": development_manifest,
        "smn4lang/story_reference.json": reference,
        "smn4lang/story_reference.provenance.json": provenance,
        "smn4lang/ovmi_support.json": support,
    }
    for size, manifest in vocabularies.items():
        documents[f"smn4lang/vocabulary_N{size}.json"] = manifest
    return documents


def _conditions_document(generator_sha256: str) -> dict:
    donor_protocol = {
        "common_support_query_set": True,
        "cross_recording_primary": False,
        "donor_word_must_differ_from_target": True,
        "donor_window_must_not_overlap_target": True,
        "same_recording": True,
        "same_subject": True,
        "same_window_contract": True,
        "seeds": list(range(20)),
        "target_and_context_position_unchanged": True,
    }
    temporal_protocol = {
        "common_support_queries": True,
        "offset_selection": "train_val_only",
        "parameter_status": "parameters_not_frozen",
        "same_recording": True,
        "same_subject": True,
        "shift": "deterministic_circular",
        "source_window_must_not_overlap_target": True,
        "target_unchanged": True,
    }
    return with_manifest_sha256(
        {
            "asset_type": "experiment_conditions",
            "candidate_vocabulary_sizes": {
                "primary": list(PRIMARY_VOCABULARY_SIZES),
                "supplementary_exploratory": [200, 300, 500],
                "test_support_may_change_sizes": False,
            },
            "conditions": {
                "neural_context.clean": {"status": "implemented"},
                "neural_context.donor_following": {"status": "exploratory"},
                "neural_context.donor_swap": {
                    "protocol": donor_protocol,
                    "status": "exploratory",
                },
                "neural_context.structure_only": {"status": "exploratory"},
                "neural_context.temporal_shift": {
                    "protocol": temporal_protocol,
                    "status": "missing",
                },
                "word.clean": {"status": "implemented"},
                "word.donor_swap": {
                    "protocol": donor_protocol,
                    "status": "missing",
                },
                "word.temporal_shift": {
                    "protocol": temporal_protocol,
                    "status": "missing",
                },
            },
            "datasets": {
                "ChineseEEG2": {
                    "confirmatory_status": "exploratory_or_replication_only",
                    "neural_context": "bounded_semantic_v1",
                    "row_context_role": "historical_structure_prior_control",
                    "test_status": "previously_used_for_exploratory_analysis",
                    "word": {"model.use_transformer": False},
                },
                "SMN4Lang": {
                    "confirmatory_status": "eligible_after_multisubject_protocol_freeze",
                    "formal_manifest_status": "awaiting_multisubject_data",
                    "neural_context": "script_sentence_then_contiguous_chunks",
                    "test_label_status": "inspected_for_support_audit",
                    "test_meg_status": "unopened",
                    "word": {"model.use_transformer": False},
                },
            },
            "domain_reference": {"status": "not_frozen"},
            "full_ovmi": {
                "missing_candidate_true_sample": "unavailable",
                "observed_support_is_not_full_ovmi": True,
                "remove_missing_candidates": False,
                "retrieval_and_coverage_remain_reportable": True,
                "smooth_zero_support_rows": False,
                "substitute_other_metric": False,
            },
            "generator": "experiments/generate_manifests.py",
            "generator_sha256": generator_sha256,
            "protocol_status": "machine_checkable_assets_frozen_without_test_meg_run",
        }
    )


def build_documents(
    chinese_event_table: Path = DEFAULT_CHINESE_EVENT_TABLE,
    smn_event_table: Path = DEFAULT_SMN_EVENT_TABLE,
) -> dict[str, dict]:
    """构造全部协议文档；这里只读取事件/文本表，不读取 EEG/MEG。"""
    generator_digest = file_sha256(Path(__file__))
    documents = {}
    documents.update(_chinese_documents(Path(chinese_event_table), generator_digest))
    documents.update(_smn_documents(Path(smn_event_table), generator_digest))
    documents["conditions.json"] = _conditions_document(generator_digest)
    return documents


def write_or_check_documents(
    documents: dict[str, dict], output_root: Path, *, check: bool
) -> list[Path]:
    """稳定写入资产，或逐字节核对现有资产。"""
    output_root = Path(output_root)
    paths = []
    mismatches = []
    for relative_path, document in sorted(documents.items()):
        path = output_root / relative_path
        expected = json_bytes(document)
        paths.append(path)
        if check:
            if not path.exists() or path.read_bytes() != expected:
                mismatches.append(relative_path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    if mismatches:
        raise ValueError(f"协议资产与当前输入不一致：{mismatches}")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只核对，不改写资产")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--chinese-event-table", type=Path, default=DEFAULT_CHINESE_EVENT_TABLE
    )
    parser.add_argument("--smn-event-table", type=Path, default=DEFAULT_SMN_EVENT_TABLE)
    args = parser.parse_args(argv)
    documents = build_documents(args.chinese_event_table, args.smn_event_table)
    paths = write_or_check_documents(documents, args.output_root, check=args.check)
    action = "已核对" if args.check else "已生成"
    print(f"{action} {len(paths)} 个协议资产：{args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
