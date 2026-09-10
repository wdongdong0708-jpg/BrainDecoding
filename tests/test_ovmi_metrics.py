"""OVMI 评估适配层的合同与回归测试。"""

import json

import numpy as np

from metrics import (
    fixed_vocabulary_retrieval,
    fixed_vocabulary_top1_predictions,
)
from ovmi_metrics import (
    build_confusion_matrix,
    full_ovmi_metrics,
    load_reference_distribution,
)


def _config(reference):
    return {
        "enabled": True,
        "method": "full",
        "reference": reference,
        "language": "zh",
    }


def _word_pairs(vocabulary, matrix):
    """把小型计数矩阵展开为评估样本，避免测试依赖内部实现。"""
    true_words = []
    predicted_words = []
    for row, true_word in zip(matrix, vocabulary):
        for count, predicted_word in zip(row, vocabulary):
            true_words.extend([true_word] * int(count))
            predicted_words.extend([predicted_word] * int(count))
    return true_words, predicted_words


def _evaluate_matrix(vocabulary, matrix, reference):
    true_words, predicted_words = _word_pairs(vocabulary, matrix)
    return full_ovmi_metrics(
        true_words,
        predicted_words,
        vocabulary,
        _config(reference),
    )


def test_perfect_predictions_create_diagonal_confusion_matrix():
    vocabulary = ["乙", "甲", "丙"]
    matrix = build_confusion_matrix(
        ["乙", "甲", "丙", "乙"],
        ["乙", "甲", "丙", "乙"],
        vocabulary,
    )
    np.testing.assert_array_equal(matrix, np.diag([2, 1, 1]))


def test_systematic_errors_reduce_full_ovmi():
    vocabulary = ["甲", "乙", "丙"]
    reference = {"甲": 3, "乙": 2, "丙": 1, "词表外": 4}
    perfect = _evaluate_matrix(vocabulary, np.diag([4, 4, 4]), reference)
    systematic = _evaluate_matrix(
        vocabulary,
        np.asarray([[0, 4, 0], [0, 4, 0], [0, 0, 4]]),
        reference,
    )
    assert perfect["available"] is True
    assert systematic["available"] is True
    assert systematic["score_bits"] < perfect["score_bits"]


def test_random_and_degenerate_predictions_have_still_lower_ovmi():
    vocabulary = ["甲", "乙", "丙"]
    reference = {"甲": 3, "乙": 2, "丙": 1}
    systematic = _evaluate_matrix(
        vocabulary,
        np.asarray([[0, 4, 0], [0, 4, 0], [0, 0, 4]]),
        reference,
    )
    random_predictions = _evaluate_matrix(
        vocabulary,
        np.ones((3, 3), dtype=np.int64),
        reference,
    )
    degenerate = _evaluate_matrix(
        vocabulary,
        np.asarray([[4, 0, 0], [4, 0, 0], [4, 0, 0]]),
        reference,
    )
    assert random_predictions["score_bits"] < systematic["score_bits"]
    assert degenerate["score_bits"] < systematic["score_bits"]


def test_full_ovmi_rejects_vocabulary_without_true_samples():
    result = full_ovmi_metrics(
        ["甲", "乙"],
        ["甲", "甲"],
        ["甲", "乙", "丙"],
        _config({"甲": 1, "乙": 1, "丙": 1}),
    )
    assert result["available"] is False
    assert result["score_bits"] is None
    assert result["missing_vocabulary_words"] == ["丙"]
    assert result["labels"] == ["甲", "乙", "丙"]


def test_reference_missing_words_are_reported():
    result = _evaluate_matrix(
        ["甲", "乙", "丙"],
        np.diag([2, 2, 2]),
        {"甲": 4, "乙": 2, "词表外": 1},
    )
    assert result["available"] is True
    assert result["reference_matched_words"] == 2
    assert result["reference_missing_words"] == 1
    assert result["reference_missing_word_labels"] == ["丙"]
    json.dumps(result, ensure_ascii=False)


def test_json_and_csv_reference_loading(tmp_path):
    json_path = tmp_path / "reference.json"
    json_path.write_text('{"甲": 3, "乙": 2}', encoding="utf-8")
    csv_path = tmp_path / "reference.csv"
    csv_path.write_text("word,count\n甲,3\n乙,2\n", encoding="utf-8")
    assert load_reference_distribution(json_path) == {"甲": 3.0, "乙": 2.0}
    assert load_reference_distribution(csv_path) == {"甲": 3.0, "乙": 2.0}


def test_top1_helper_does_not_change_existing_top1_or_top10_metrics():
    target_words = ["alpha", "beta", "gamma", "alpha"]
    target_embeddings = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
        dtype=np.float32,
    )
    query_embeddings = np.asarray(
        [[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]],
        dtype=np.float32,
    )
    vocabulary = ["beta", "alpha", "gamma"]

    before, _, _ = fixed_vocabulary_retrieval(
        query_embeddings,
        target_embeddings,
        target_words,
        vocabulary,
        top_ks=(1, 10),
        vocabulary_name="test3",
    )
    true_words, predicted_words, selected = fixed_vocabulary_top1_predictions(
        query_embeddings,
        target_embeddings,
        target_words,
        vocabulary,
    )
    after, _, _ = fixed_vocabulary_retrieval(
        query_embeddings,
        target_embeddings,
        target_words,
        vocabulary,
        top_ks=(1, 10),
        vocabulary_name="test3",
    )

    assert before == after
    assert before["retrieval_acc1_vocab=test3"] == 0.5
    assert before["retrieval_acc10_vocab=test3"] == 1.0
    assert true_words == target_words
    assert predicted_words == ["alpha", "gamma", "gamma", "beta"]
    np.testing.assert_array_equal(selected, np.arange(4))
