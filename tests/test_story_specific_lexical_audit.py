"""Story-specific lexical audit v0 mathematical and safety contracts."""

import inspect
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from braindecoding.evaluation import story_specific as audit


def _chinese_row(subject, split, word, source_word_id, *, trainable=True):
    return {
        "subject_id": subject,
        "split": split,
        "is_trainable": trainable,
        "normalized_word": word,
        "voice_version": "f1",
        "source_word_id": source_word_id,
        "chapter": 1 if split == "train" else 2,
        "chapter_word_index": source_word_id,
    }


def _math_result(alpha=0.5):
    return audit.score_story_distribution(
        {"a": 3, "b": 1},
        {"a": 1, "b": 3, "c": 2},
        alpha=alpha,
    )


def test_multi_subject_repetition_is_not_counted_as_story_frequency():
    table = pd.DataFrame(
        [
            _chinese_row("sub-01", "train", "小王子", 1),
            _chinese_row("sub-02", "train", "小王子", 1),
            _chinese_row("sub-01", "val", "狐狸", 2),
            _chinese_row("sub-02", "val", "狐狸", 2),
        ]
    )
    train, full = audit._chinese_material_views(table)
    assert train["_protocol_word"].tolist() == ["小王子"]
    assert full["_protocol_word"].tolist() == ["小王子", "狐狸"]


def test_train_ranking_does_not_read_val_or_test_material():
    train_counts = {"甲": 3, "乙": 2}
    background = {"甲": 4, "乙": 1, "测试词": 1}
    before = audit.score_story_distribution(
        train_counts, background, alpha=0.5, train_counts=train_counts
    )
    audit.score_story_distribution(
        {"甲": 3, "乙": 2, "测试词": 1000},
        background,
        alpha=0.5,
        train_counts=train_counts,
    )
    after = audit.score_story_distribution(
        train_counts, background, alpha=0.5, train_counts=train_counts
    )
    assert before["rows"] == after["rows"]


def test_full_story_reference_is_separate_from_train_candidate_ranking():
    train = audit.score_story_distribution(
        {"train": 4, "shared": 1},
        {"train": 1, "shared": 2, "full": 2},
        alpha=1.0,
    )
    full = audit.score_story_distribution(
        {"train": 4, "shared": 1, "full": 50},
        {"train": 1, "shared": 2, "full": 2},
        alpha=1.0,
        train_counts={"train": 4, "shared": 1},
    )
    assert train["rows"][0]["word"] != full["rows"][0]["word"]
    assert [row["word"] for row in train["rows"]] == ["train", "shared"]


def test_chinese_normalization_keeps_existing_standard_word_surface():
    assert audit.normalize_chinese_background_word(" 小王子 ") == "小王子"
    assert audit.normalize_chinese_background_word("狐狸") == "狐狸"


def test_french_normalization_preserves_accents():
    assert audit.normalize_french_background_word(" ÉTOILE ") == "étoile"


def test_french_normalization_does_not_merge_source_tokens():
    assert [
        audit.normalize_french_background_word(word) for word in ("j", "avais")
    ] == ["j", "avais"]


def test_background_loader_directly_reuses_ovmi_reader(tmp_path, monkeypatch):
    path = tmp_path / "background.json"
    path.write_text("{}", encoding="utf-8")
    calls = []

    def fake_loader(received):
        calls.append(Path(received))
        return {" ÉTÉ ": 2.0, "été": 3.0}

    monkeypatch.setattr(audit, "load_reference_distribution", fake_loader)
    counts, metadata = audit.load_background_reference(
        path, normalize=audit.normalize_french_background_word, language="fr"
    )
    assert calls == [path.resolve()]
    assert counts == {"été": 5.0}
    assert metadata["reader"].endswith("ovmi.load_reference_distribution")


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_background_smoothing_matches_candidate_v0_formula(alpha):
    result = _math_result(alpha)
    rows = {row["word"]: row for row in result["rows"]}
    denominator = 6 + alpha * 3
    assert rows["a"]["p_background"] == pytest.approx((1 + alpha) / denominator)
    assert rows["b"]["p_background"] == pytest.approx((3 + alpha) / denominator)
    assert result["universe_type_count"] == 3


def test_raw_log_lift_math_is_correct():
    rows = {row["word"]: row for row in _math_result()["rows"]}
    expected = math.log2(0.75 / (1.5 / 7.5))
    assert rows["a"]["raw_log_lift_bits"] == pytest.approx(expected)


def test_signed_kl_contribution_math_is_correct():
    rows = {row["word"]: row for row in _math_result()["rows"]}
    assert rows["a"]["signed_kl_contribution_bits"] == pytest.approx(
        0.75 * rows["a"]["raw_log_lift_bits"]
    )


def test_positive_score_truncates_negative_lift():
    rows = {row["word"]: row for row in _math_result()["rows"]}
    assert rows["b"]["raw_log_lift_bits"] < 0
    assert rows["b"]["signed_kl_contribution_bits"] < 0
    assert rows["b"]["positive_story_specific_score_bits"] == 0


def test_signed_contribution_sum_sanity_check():
    result = _math_result()
    assert result["signed_kl_sum_sanity_check"] is True
    assert result["total_kl_bits"] == pytest.approx(
        result["signed_contribution_sum_bits"]
    )
    assert result["sum_positive_story_specific_score_bits"] > result["total_kl_bits"]


def test_ranking_is_deterministic():
    args = ({"b": 2, "a": 2}, {"b": 1, "a": 1})
    first = audit.score_story_distribution(*args, alpha=0.5)
    second = audit.score_story_distribution(*args, alpha=0.5)
    assert first["rows"] == second["rows"]


def test_tie_break_is_count_then_word_ascending():
    result = audit.score_story_distribution(
        {"b": 2, "a": 2}, {"b": 1, "a": 1}, alpha=1.0
    )
    assert [row["word"] for row in result["rows"]] == ["a", "b"]


def test_threshold_uses_train_material_count_only():
    result = audit.score_story_distribution(
        {"high_full_low_train": 100, "train_supported": 5},
        {"high_full_low_train": 1, "train_supported": 1},
        alpha=0.5,
        train_counts={"high_full_low_train": 1, "train_supported": 5},
    )
    sensitivity = audit.threshold_sensitivity(result["rows"])
    assert sensitivity["thresholds"]["5"]["top50_words"] == ["train_supported"]


def test_threshold_eligibility_remains_available_without_background():
    eligibility = audit.threshold_eligibility({"a": 20, "b": 5, "c": 1})
    assert eligibility["1"]["eligible_type_count"] == 3
    assert eligibility["5"]["eligible_type_count"] == 2
    assert eligibility["10"]["eligible_type_count"] == 1
    assert eligibility["1"]["top50_words"] is None


def test_background_missing_and_suspicious_flags_are_correct():
    flags = audit.diagnostic_flags("-1909", True, 1)
    assert flags == {
        "background_missing": True,
        "number_like": True,
        "punctuation_only": False,
        "contains_digit": True,
        "leading_hyphen": True,
        "single_character_or_token": False,
        "very_low_train_count": True,
    }
    assert audit.diagnostic_flags("!", False, 8)["punctuation_only"] is True
    assert audit.diagnostic_flags("j", False, 8)["single_character_or_token"] is True


def test_suspicious_tokens_are_flagged_but_not_removed():
    result = audit.score_story_distribution(
        {"j": 2, "-là": 1, "1909": 1, "!": 1},
        {"j": 4},
        alpha=0.5,
    )
    assert {row["word"] for row in result["rows"]} == {"j", "-là", "1909", "!"}


def test_surface_diagnostics_do_not_require_background_or_delete_tokens():
    result = audit.lexical_surface_diagnostics(
        {"j": 5, "-là": 2, "1909": 1, "mot": 3},
        {"j": 4, "-là": 1, "1909": 0, "mot": 2},
    )
    words = {
        row["word"] for row in result["potentially_suspicious_surface_tokens"]
    }
    assert words == {"-là", "1909"}
    assert result["automatic_deletion_performed"] is False
    assert "j" in result["single_character_examples_by_frequency"]


def test_coverage_reports_token_types_missing_and_top100_without_gate():
    result = audit.coverage_audit({"a": 5, "b": 2, "c": 1}, {"a": 10, "c": 1})
    assert result["matched_token_count"] == 6
    assert result["total_token_count"] == 8
    assert result["matched_token_fraction"] == 0.75
    assert result["matched_type_count"] == 2
    assert result["background_missing_types"] == ["b"]
    assert result["top100_story_frequency_background_missing"] == ["b"]
    assert result["coverage_pass_threshold"] is None


def test_smoothing_comparison_reports_all_requested_sizes():
    half = audit.score_story_distribution(
        {str(index): index + 1 for index in range(160)},
        {str(index): 2 * index + 1 for index in range(160)},
        alpha=0.5,
    )
    one = audit.score_story_distribution(
        {str(index): index + 1 for index in range(160)},
        {str(index): 2 * index + 1 for index in range(160)},
        alpha=1.0,
    )
    comparison = audit.ranking_comparison(half["rows"], one["rows"])
    assert {f"top{size}" for size in audit.TOP_SIZES}.issubset(comparison)
    assert comparison["sensitivity_flag_rule"].endswith("not a confirmatory gate")


def _fake_material(tmp_path):
    event = tmp_path / "events.csv"
    story = tmp_path / "story_reference.json"
    event.write_text("lexical fixture only\n", encoding="utf-8")
    story.write_text('{"a": 3, "b": 2}', encoding="utf-8")
    return {
        "train_counts": {"a": 2, "b": 1},
        "full_counts": {"a": 3, "b": 2},
        "train_units": ["train-unit"],
        "full_units": ["full-unit"],
        "material_source": {"subject_repetitions_counted": False},
        "event_table": {"path": str(event), "sha256": audit.file_sha256(event)},
        "story_reference": {
            "path": str(story),
            "sha256": audit.file_sha256(story),
            "verified_equal_to_full_story_counts": True,
        },
        "canonical_assets": {},
    }


def test_completed_manifest_is_exploratory_and_outputs_only_under_reports(
    tmp_path, monkeypatch
):
    report_root = tmp_path / "reports" / audit.AUDIT_NAME
    monkeypatch.setattr(audit, "REPORT_ROOT", report_root)
    monkeypatch.setattr(audit, "load_story_counts", lambda dataset: _fake_material(tmp_path))
    background = tmp_path / "background.json"
    background.write_text('{"a": 2, "b": 4, "ordinary": 8}', encoding="utf-8")
    output, manifest = audit.run_dataset_audit(
        "chineseeeg2_littleprince", background=background
    )
    assert output.is_relative_to(report_root)
    assert manifest["status"] == "completed"
    assert manifest["exploratory_only"] is True
    assert manifest["formula_status"] == "candidate_v0_not_frozen"
    assert manifest["test_neural_data_opened"] is False
    assert manifest["test_model_evaluation"] == "not_run"
    assert manifest["test_predictions_generated"] is False
    assert manifest["test_metrics_inspected"] is False
    assert (output / "train_word_scores_alpha0p5.csv").is_file()
    assert (output / "full_story_word_scores_alpha1p0.csv").is_file()
    assert (output / "story_specific_reference_candidate_v0.json").is_file()
    assert not (tmp_path / "experiments").exists()


def test_blocked_audit_does_not_modify_frozen_vocabularies_or_story_references(
    tmp_path, monkeypatch
):
    paths = [
        Path(audit.DATASETS[dataset]["manifest_dir"]) / name
        for dataset in audit.DATASETS
        for name in (
            "vocabulary_N20.json",
            "vocabulary_N50.json",
            "vocabulary_N100.json",
            "vocabulary_N150.json",
            "story_reference.json",
        )
    ]
    before = {path: audit.file_sha256(path) for path in paths}
    monkeypatch.setattr(audit, "REPORT_ROOT", tmp_path / "reports" / audit.AUDIT_NAME)
    for dataset in audit.DATASETS:
        _, manifest = audit.run_dataset_audit(dataset)
        assert manifest["status"] == "blocked"
        assert manifest["reason"] == "background_reference_not_provided"
    assert {path: audit.file_sha256(path) for path in paths} == before


def test_module_has_no_checkpoint_model_forward_or_neural_array_reader():
    source = inspect.getsource(audit)
    assert "np.load(" not in source
    assert "torch.load(" not in source
    assert ".forward(" not in source
    assert "read_raw_brainvision" not in source
    assert "read_raw_fif" not in source
    assert "model(" not in source


def test_default_output_directory_is_reports_only():
    for dataset in audit.DATASETS:
        path = audit.dataset_output_directory(dataset).resolve()
        assert (audit.PROJECT_ROOT / "reports").resolve() in path.parents
        assert (audit.PROJECT_ROOT / "experiments").resolve() not in path.parents


def test_blocked_manifest_explicitly_records_not_frozen_and_no_test_evaluation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "REPORT_ROOT", tmp_path / "reports" / audit.AUDIT_NAME)
    monkeypatch.setattr(audit, "load_story_counts", lambda dataset: _fake_material(tmp_path))
    output, manifest = audit.run_dataset_audit("pallier2025")
    on_disk = json.loads((output / "exploratory_manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["formula_status"] == "candidate_v0_not_frozen"
    assert on_disk["checkpoint_opened"] is False
    assert on_disk["test_neural_data_opened"] is False
    assert on_disk["test_model_evaluation"] == "not_run"
