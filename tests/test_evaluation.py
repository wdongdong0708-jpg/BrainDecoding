"""公共词表、检索和 OVMI 评价边界的行为锁定测试。"""

import json

import numpy as np
import pytest

from braindecoding.evaluation import ovmi, retrieval, vocabulary


def _embeddings():
    target_words = ["alpha", "beta", "gamma", "alpha"]
    target_embeddings = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
        dtype=np.float32,
    )
    query_embeddings = np.asarray(
        [[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]],
        dtype=np.float32,
    )
    return query_embeddings, target_embeddings, target_words


def test_retrieval_ranks_and_ties_are_unchanged():
    query_embeddings = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    candidate_embeddings = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    candidate_ids = ["a", "b", "c"]
    target_ids = ["a", "c", "a"]

    actual = retrieval.retrieval_ranks(
        query_embeddings,
        target_ids,
        candidate_embeddings,
        candidate_ids,
        chunk_size=1,
    )
    np.testing.assert_array_equal(actual, [1.5, 1.0, 2.0])

    actual_summary = retrieval.summarize_retrieval(
        actual, target_ids, candidate_count=3, top_ks=(1, 2)
    )
    assert actual_summary["micro_recall_at_1"] == pytest.approx(1 / 3)
    assert actual_summary["macro_recall_at_1"] == pytest.approx(0.5)


def test_fixed_vocabulary_predictions_and_metrics_are_unchanged():
    query_embeddings, target_embeddings, target_words = _embeddings()
    candidates = ["beta", "alpha", "gamma"]

    actual_predictions = retrieval.fixed_vocabulary_top1_predictions(
        query_embeddings, target_embeddings, target_words, candidates
    )
    assert actual_predictions[:2] == (
        ["alpha", "beta", "gamma", "alpha"],
        ["alpha", "gamma", "gamma", "beta"],
    )
    np.testing.assert_array_equal(actual_predictions[2], [0, 1, 2, 3])

    actual_metrics = retrieval.fixed_vocabulary_retrieval_metrics(
        query_embeddings,
        target_embeddings,
        target_words,
        candidates,
        vocabulary_name="test3",
    )
    assert actual_metrics["retrieval_acc1_vocab=test3"] == 0.5
    assert actual_metrics["retrieval_acc10_vocab=test3"] == 1.0


def test_confusion_full_ovmi_and_coverage_are_unchanged():
    words = ["甲", "乙", "丙"]
    true_words = ["甲", "甲", "乙", "乙", "丙", "丙"]
    predicted_words = ["甲", "乙", "乙", "乙", "丙", "甲"]
    reference = {"甲": 5, "乙": 3, "丙": 2, "词表外": 1}
    config = {
        "enabled": True,
        "method": "full",
        "reference": reference,
        "language": "zh",
    }

    actual_matrix = ovmi.build_confusion_matrix(true_words, predicted_words, words)
    np.testing.assert_array_equal(actual_matrix, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])

    actual = ovmi.full_ovmi_metrics(true_words, predicted_words, words, config)
    assert actual["score_bits"] == pytest.approx(0.5787877108333521)
    assert actual["coverage"] == pytest.approx(0.9090909090909091)
    assert actual["in_vocab_information_bits"] == pytest.approx(0.6366664819166874)
    assert actual["output_entropy_bits"] == pytest.approx(1.3366664819166874)
    assert actual["conditional_entropy_bits"] == pytest.approx(0.7)


def test_reference_mapping_json_and_csv_loading_are_unchanged(tmp_path):
    mapping = {"甲": 3, "乙": 2}
    json_path = tmp_path / "reference.json"
    json_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    csv_path = tmp_path / "reference.csv"
    csv_path.write_text("word,count\n甲,3\n乙,2\n", encoding="utf-8")

    for source in (mapping, json_path, csv_path):
        assert ovmi.load_reference_distribution(source) == mapping


def test_missing_required_reference_is_explicitly_unavailable(tmp_path):
    missing = tmp_path / "missing-reference.json"
    result = ovmi.full_ovmi_metrics(
        ["甲", "乙"],
        ["甲", "乙"],
        ["甲", "乙"],
        {
            "enabled": True,
            "method": "full",
            "reference": str(missing),
            "language": "zh",
        },
    )
    assert result["available"] is False
    assert result["reason"] == "reference_or_official_ovmi_unavailable"
    assert result["score_bits"] is None
    assert missing.name in result["error"]


def test_frequency_vocabulary_is_deterministic_for_duplicates_and_ties():
    words = [" Beta ", "alpha", "BETA", "ALPHA", "gamma", "delta"]
    normalizer = lambda value: str(value).strip().lower()
    expected = ("alpha", "beta", "delta")
    assert vocabulary.build_frequency_vocabulary(
        words, size=3, normalize=normalizer
    ) == expected
    assert vocabulary.build_frequency_vocabulary(
        reversed(words), size=3, normalize=normalizer
    ) == expected


def test_vocabulary_builder_only_reads_explicit_words():
    class ExplicitWords:
        @property
        def test(self):
            raise AssertionError("词表构建器不应读取 test 数据。")

        def __iter__(self):
            return iter(["甲", "乙", "甲"])

    assert vocabulary.build_frequency_vocabulary(ExplicitWords(), 2) == ("甲", "乙")


def test_frozen_vocabulary_validation_and_metadata():
    frozen = vocabulary.validate_frozen_vocabulary(
        [" Alpha ", "beta"],
        expected_size=2,
        normalize=lambda value: str(value).strip().lower(),
    )
    assert frozen == ("alpha", "beta")
    with pytest.raises(ValueError, match="重复"):
        vocabulary.validate_frozen_vocabulary(
            ["Alpha", "alpha"], normalize=lambda value: str(value).lower()
        )

    metadata = vocabulary.build_vocabulary_metadata(
        frozen,
        source_split="train",
        policy="frequency_desc_word_asc",
        normalization="lower_strip",
        word_counts={"alpha": 4, "beta": 3, "unused": 2},
    )
    assert metadata == {
        "vocabulary": ["alpha", "beta"],
        "size": 2,
        "source_split": "train",
        "policy": "frequency_desc_word_asc",
        "normalization": "lower_strip",
        "word_counts": {"alpha": 4, "beta": 3},
    }


def test_story_reference_counts_only_explicit_words_and_is_independent():
    source_words = ["故事", "词", "故事", ""]
    reference = ovmi.build_reference_distribution(source_words)
    assert reference == {"故事": 2, "词": 1}

    candidate = vocabulary.build_frequency_vocabulary(["甲", "甲", "乙"], 2)
    unrelated_reference = ovmi.build_reference_distribution(["丙"] * 100)
    assert candidate == ("甲", "乙")
    assert unrelated_reference == {"丙": 100}


def test_word_evaluators_import_official_common_ovmi():
    from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import evaluate as chinese
    from braindecoding.tasks.word_decoding.libribrain100 import evaluate as libri
    from braindecoding.tasks.word_decoding.smn4lang import evaluate as smn

    for module in (chinese, smn, libri):
        assert module.fixed_vocabulary_ovmi_metrics is ovmi.fixed_vocabulary_ovmi_metrics


def test_word_training_evaluators_reach_the_same_retrieval_implementation():
    from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import train as chinese
    from braindecoding.tasks.word_decoding.libribrain100 import train as libri
    from braindecoding.tasks.word_decoding.smn4lang import train as smn

    for module in (chinese, smn, libri):
        assert (
            module.fixed_vocabulary_retrieval_metrics
            is retrieval.fixed_vocabulary_retrieval_metrics
        )
