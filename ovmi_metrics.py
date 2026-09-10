"""显式评估阶段使用的 OVMI 指标适配层。"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from metrics import fixed_vocabulary_top1_predictions


def _ordered_labels(vocabulary):
    """保留固定词表顺序，同时沿用现有检索的不区分大小写规则。"""
    labels = [str(word).lower() for word in vocabulary]
    if len(labels) != len(set(labels)):
        raise ValueError("固定评价词表在不区分大小写后包含重复词。")
    return labels


def build_confusion_matrix(true_words, predicted_words, vocabulary):
    """按固定词表原始顺序构造 true_word × predicted_word 计数矩阵。"""
    labels = _ordered_labels(vocabulary)
    true_words = [str(word).lower() for word in true_words]
    predicted_words = [str(word).lower() for word in predicted_words]
    if len(true_words) != len(predicted_words):
        raise ValueError("真实词和预测词数量不一致。")

    positions = {word: index for index, word in enumerate(labels)}
    unknown_true = [word for word in true_words if word not in positions]
    unknown_predicted = [word for word in predicted_words if word not in positions]
    if unknown_true or unknown_predicted:
        raise ValueError(
            "混淆矩阵输入包含固定词表外的词："
            f"true={sorted(set(unknown_true))}, "
            f"predicted={sorted(set(unknown_predicted))}"
        )

    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_word, predicted_word in zip(true_words, predicted_words):
        matrix[positions[true_word], positions[predicted_word]] += 1
    return matrix


def _validated_reference(reference):
    """把文件内容规范为官方 OVMI 接受的非负 word -> count 映射。"""
    if not isinstance(reference, Mapping) or not reference:
        raise ValueError("reference 必须是非空的 word -> count 映射。")
    result = {}
    for raw_word, raw_count in reference.items():
        word = str(raw_word).strip()
        if not word:
            raise ValueError("reference 包含空词。")
        count = float(raw_count)
        if not np.isfinite(count) or count < 0:
            raise ValueError(f"reference 词频必须有限且非负：{word}")
        result[word] = result.get(word, 0.0) + count
    if sum(result.values()) <= 0:
        raise ValueError("reference 总词频必须大于零。")
    return result


def _load_csv_reference(path):
    """读取带 word/count 列的公共参考词频 CSV。"""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = {str(name).strip().lower(): name for name in reader.fieldnames or ()}
        word_column = fields.get("word")
        count_column = next(
            (
                fields[name]
                for name in ("count", "frequency", "freq", "freqcount")
                if name in fields
            ),
            None,
        )
        if word_column is None or count_column is None:
            raise ValueError(
                "reference CSV 必须包含 word 列和 count/frequency/freq/FreqCount 列。"
            )
        reference = {}
        for row in reader:
            word = str(row[word_column]).strip()
            if not word:
                continue
            count = float(row[count_column])
            reference[word] = reference.get(word, 0.0) + count
    return _validated_reference(reference)


def load_reference_distribution(reference, base_dir=None):
    """加载官方 SUBTLEX-UK 默认参考，或外部 JSON/CSV 词频文件。"""
    if isinstance(reference, Mapping):
        return _validated_reference(reference)
    if reference is None or str(reference).strip().lower() in {
        "subtlex_uk",
        "subtlex-uk",
        "default",
    }:
        from ovmi import load_subtlex_uk

        return load_subtlex_uk()

    path = Path(reference)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    path = path.resolve()
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            return _validated_reference(json.load(file))
    if path.suffix.lower() == ".csv":
        return _load_csv_reference(path)
    raise ValueError("reference 只支持 JSON 或 CSV 文件。")


def _reference_name(reference):
    if isinstance(reference, Mapping):
        return "inline"
    if reference is None:
        return "subtlex_uk"
    return str(reference)


def _result_template(config, vocabulary):
    """保证可用与不可用结果具有同一组稳定、可序列化字段。"""
    reference = config.get("reference", "subtlex_uk")
    return {
        "available": False,
        "method": str(config.get("method", "full")).lower(),
        "reference": _reference_name(reference),
        "language": str(config.get("language", "")),
        "score_bits": None,
        "coverage": None,
        "in_vocab_information_bits": None,
        "output_entropy_bits": None,
        "conditional_entropy_bits": None,
        "vocabulary_size": int(len(vocabulary)),
        "reference_matched_words": None,
        "reference_missing_words": None,
        "reference_matched_word_labels": [],
        "reference_missing_word_labels": [],
        "missing_vocabulary_words": [],
    }


def full_ovmi_metrics(
    true_words,
    predicted_words,
    vocabulary,
    config,
    base_dir=None,
):
    """用官方 OVMI full 模式计算并返回 JSON 可序列化明细。"""
    labels = _ordered_labels(vocabulary)
    result = _result_template(config, labels)
    method = str(config.get("method", "full")).lower()
    if method != "full":
        raise ValueError("当前主 OVMI 指标只允许 method='full'。")

    matrix = build_confusion_matrix(true_words, predicted_words, labels)
    result["labels"] = labels
    result["confusion_matrix"] = matrix.tolist()
    missing_vocabulary_words = [
        labels[index] for index, count in enumerate(matrix.sum(axis=1)) if count == 0
    ]
    result["missing_vocabulary_words"] = missing_vocabulary_words
    if missing_vocabulary_words:
        result["reason"] = "full_ovmi_requires_true_samples_for_every_word"
        return result

    reference_spec = config.get("reference", "subtlex_uk")
    try:
        reference = load_reference_distribution(reference_spec, base_dir=base_dir)
        from ovmi import ovmi
    except (ImportError, OSError) as error:
        result["reason"] = "reference_or_official_ovmi_unavailable"
        result["error"] = str(error)
        return result

    reference_words = set(reference)
    matched_reference_words = [word for word in labels if word in reference_words]
    missing_reference_words = [word for word in labels if word not in reference_words]
    result["reference_matched_words"] = int(len(matched_reference_words))
    result["reference_missing_words"] = int(len(missing_reference_words))
    result["reference_matched_word_labels"] = matched_reference_words
    result["reference_missing_word_labels"] = missing_reference_words

    details = ovmi(
        reference,
        labels,
        method="full",
        confusion_matrix=matrix,
        labels=labels,
        return_details=True,
    )
    result.update(
        {
            "available": True,
            "score_bits": float(details.score),
            "coverage": float(details.coverage),
            "in_vocab_information_bits": float(details.in_vocab_information),
            "output_entropy_bits": float(details.output_entropy),
            "conditional_entropy_bits": float(details.conditional_entropy),
            "vocabulary_size": int(details.vocabulary_size),
        }
    )
    return result


def fixed_vocabulary_ovmi_metrics(
    query_embeddings,
    target_embeddings,
    target_words,
    vocabulary,
    config,
    base_dir=None,
):
    """从现有固定词表余弦 Top-1 预测计算 full OVMI。"""
    labels = _ordered_labels(vocabulary)
    result = _result_template(config, labels)
    if not bool(config.get("enabled", False)):
        result["reason"] = "disabled"
        return result

    true_words, predicted_words, _ = fixed_vocabulary_top1_predictions(
        query_embeddings,
        target_embeddings,
        target_words,
        labels,
    )
    return full_ovmi_metrics(
        true_words,
        predicted_words,
        labels,
        config,
        base_dir=base_dir,
    )
