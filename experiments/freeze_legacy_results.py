"""冻结当前历史科学运行文件的不可变 SHA-256 清单。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from braindecoding.config import PROJECT_ROOT
from braindecoding.experiment import file_sha256


def _identity(task, dataset, subject_scope, experiment_id, seed=0):
    return {
        "task": task,
        "dataset": dataset,
        "subject_scope": subject_scope,
        "experiment_id": experiment_id,
        "seed": seed,
    }


def _run(legacy_run_id, legacy_config, legacy_output, canonical_identity=None):
    record = {
        "legacy_run_id": legacy_run_id,
        "legacy_config": legacy_config,
        "legacy_output": legacy_output,
        "canonical_identity": canonical_identity,
        "status": (
            "reproducible_legacy" if canonical_identity else "archive_only"
        ),
    }
    if canonical_identity is None:
        record["reason"] = "exact_config_unavailable"
    return record


LEGACY_RUNS = (
    _run(
        "chineseeeg1_sr_closed_set_loso",
        "configs/ChineseEEG_SR.yaml",
        "outputs/ChineseEEG1_SR/LittlePrince_closed_set_loso",
        _identity(
            "sequence_decoding",
            "chineseeeg1_sr",
            "sub04-10_sub13-14",
            "historical_closed_set_loso",
            42,
        ),
    ),
    _run(
        "chineseeeg1_sr_fourier_subject_row_retrieval",
        "configs/ChineseEEG_SR.yaml",
        "outputs/ChineseEEG1_SR/LittlePrince_fourier_subject_row_retrieval",
        _identity(
            "sequence_decoding",
            "chineseeeg1_sr",
            "sub04-10_sub13-14",
            "historical_row_retrieval_fourier_subject",
            42,
        ),
    ),
    _run(
        "chineseeeg1_sr_row_retrieval_unknown_config",
        None,
        "outputs/ChineseEEG1_SR/LittlePrince_row_retrieval",
    ),
    _run(
        "chineseeeg2_sub01_word",
        "configs/ChineseEEG2_LittlePrince_sub01_actual_reading_1s_cnn_only.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01/actual_reading_1s/cnn_only",
        _identity("word_decoding", "chineseeeg2_littleprince", "sub01", "scaling_word"),
    ),
    _run(
        "chineseeeg2_sub01_context_row",
        "configs/ChineseEEG2_LittlePrince_sub01_actual_reading_1s_cnn_warm_start.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01/actual_reading_1s/cnn_warm_start",
        _identity(
            "word_decoding",
            "chineseeeg2_littleprince",
            "sub01",
            "scaling_context_row",
        ),
    ),
    _run(
        "chineseeeg2_sub01_02_word",
        "configs/ChineseEEG2_LittlePrince_sub01_sub02_actual_reading_1s_cnn_only.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01_sub-02/actual_reading_1s/cnn_only",
        _identity("word_decoding", "chineseeeg2_littleprince", "sub01-02", "scaling_word"),
    ),
    _run(
        "chineseeeg2_sub01_02_context_row",
        "configs/ChineseEEG2_LittlePrince_sub01_sub02_actual_reading_1s_cnn_warm_start.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01_sub-02/actual_reading_1s/cnn_warm_start",
        _identity(
            "word_decoding",
            "chineseeeg2_littleprince",
            "sub01-02",
            "scaling_context_row",
        ),
    ),
    _run(
        "chineseeeg2_sub01_04_context_scratch",
        "configs/ChineseEEG2_LittlePrince_sub01_sub04_actual_reading_1s.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-04/actual_reading_1s/base",
        _identity(
            "word_decoding",
            "chineseeeg2_littleprince",
            "sub01-04",
            "ablation_context_scratch",
        ),
    ),
    _run(
        "chineseeeg2_sub01_04_word",
        "configs/ChineseEEG2_LittlePrince_sub01_sub04_actual_reading_1s_cnn_only.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-04/actual_reading_1s/cnn_only",
        _identity("word_decoding", "chineseeeg2_littleprince", "sub01-04", "scaling_word"),
    ),
    _run(
        "chineseeeg2_sub01_04_context_row",
        "configs/ChineseEEG2_LittlePrince_sub01_sub04_actual_reading_1s_cnn_warm_start.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-04/actual_reading_1s/cnn_warm_start",
        _identity(
            "word_decoding",
            "chineseeeg2_littleprince",
            "sub01-04",
            "scaling_context_row",
        ),
    ),
    _run(
        "chineseeeg2_sub01_08_word",
        "configs/ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_cnn_only.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-08/actual_reading_1s/cnn_only",
        _identity("word_decoding", "chineseeeg2_littleprince", "sub01-08", "main_word"),
    ),
    _run(
        "chineseeeg2_sub01_08_context_row",
        "configs/ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_cnn_warm_start.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-08/actual_reading_1s/cnn_warm_start",
        _identity(
            "word_decoding",
            "chineseeeg2_littleprince",
            "sub01-08",
            "ablation_context_row",
        ),
    ),
    _run(
        "chineseeeg2_sub01_08_context_semantic",
        "configs/ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_semantic_cnn_warm_start.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-08/actual_reading_1s/semantic_context_cnn_warm_start",
        _identity("word_decoding", "chineseeeg2_littleprince", "sub01-08", "main_context"),
    ),
    _run(
        "chineseeeg2_sub05_08_word",
        "configs/ChineseEEG2_LittlePrince_sub05_sub08_actual_reading_1s_cnn_only.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-05_to_sub-08/actual_reading_1s/cnn_only",
        _identity("word_decoding", "chineseeeg2_littleprince", "sub05-08", "scaling_word"),
    ),
    _run(
        "chineseeeg2_sub05_08_context_row",
        "configs/ChineseEEG2_LittlePrince_sub05_sub08_actual_reading_1s_cnn_warm_start.yaml",
        "outputs/ChineseEEG2_LittlePrince/sub-05_to_sub-08/actual_reading_1s/cnn_warm_start",
        _identity(
            "word_decoding",
            "chineseeeg2_littleprince",
            "sub05-08",
            "scaling_context_row",
        ),
    ),
    _run(
        "libribrain100_grouped_3s",
        "configs/LibriBrain100.yaml",
        "outputs/LibriBrain100/word_decoding",
        _identity("word_decoding", "libribrain100", "sub0", "historical_grouped_3s"),
    ),
    _run(
        "libribrain100_grouped_warm_start_1s",
        "configs/LibriBrain100_1s_cnn_warm_start.yaml",
        "outputs/LibriBrain100/word_decoding_1s_cnn_warm_start",
        _identity(
            "word_decoding",
            "libribrain100",
            "sub0",
            "historical_grouped_warm_start_1s",
        ),
    ),
    _run(
        "libribrain100_word_1s",
        "configs/LibriBrain100_1s_conv_only.yaml",
        "outputs/LibriBrain100/word_decoding_1s_conv_only",
        _identity("word_decoding", "libribrain100", "sub0", "historical_word_1s"),
    ),
    _run(
        "libribrain100_grouped_1s",
        "configs/LibriBrain100_1s.yaml",
        "outputs/LibriBrain100/word_decoding_1s_sentence_transformer",
        _identity("word_decoding", "libribrain100", "sub0", "historical_grouped_1s"),
    ),
    _run(
        "libribrain100_singleton_1s",
        "configs/LibriBrain100_1s_single_word_transformer.yaml",
        "outputs/LibriBrain100/word_decoding_1s_single_word_transformer",
        _identity("word_decoding", "libribrain100", "sub0", "historical_singleton_1s"),
    ),
    _run(
        "libribrain100_singleton_3s",
        "configs/LibriBrain100_3s_single_word_transformer.yaml",
        "outputs/LibriBrain100/word_decoding_3s_single_word_transformer",
        _identity("word_decoding", "libribrain100", "sub0", "historical_singleton_3s"),
    ),
    _run(
        "smn4lang_unknown_word_decoding",
        None,
        "outputs/SMN4Lang/word_decoding",
    ),
    _run(
        "smn4lang_gpt2_context",
        "configs/SMN4Lang_gpt2.yaml",
        "outputs/SMN4Lang/word_decoding_all_words_gpt2_layer24_1024_6400updates",
        _identity("word_decoding", "smn4lang", "sub01", "ablation_text_gpt2_context"),
    ),
    _run(
        "smn4lang_gpt2_word",
        "configs/SMN4Lang_gpt2_conv_only.yaml",
        "outputs/SMN4Lang/word_decoding_all_words_gpt2_layer24_1024_conv_only_6400updates",
        _identity("word_decoding", "smn4lang", "sub01", "ablation_text_gpt2_word"),
    ),
    _run(
        "smn4lang_mengzi_context_warm_start_1s",
        "configs/SMN4Lang_1s_cnn_warm_start.yaml",
        "outputs/SMN4Lang/word_decoding_all_words_mengzi_1s_cnn_warm_start_freeze960_6400updates",
        _identity("word_decoding", "smn4lang", "sub01", "ablation_context_warm_start"),
    ),
    _run(
        "smn4lang_mengzi_word_1s",
        "configs/SMN4Lang_1s_conv_only.yaml",
        "outputs/SMN4Lang/word_decoding_all_words_mengzi_1s_conv_only_6400updates",
        _identity("word_decoding", "smn4lang", "sub01", "dev_word"),
    ),
    _run(
        "smn4lang_mengzi_context_1s",
        "configs/SMN4Lang_1s.yaml",
        "outputs/SMN4Lang/word_decoding_all_words_mengzi_1s_sentence_transformer_6400updates",
        _identity("word_decoding", "smn4lang", "sub01", "dev_context"),
    ),
    _run(
        "smn4lang_mengzi_singleton_1s",
        "configs/SMN4Lang_1s_single_word_transformer.yaml",
        "outputs/SMN4Lang/word_decoding_all_words_mengzi_1s_single_word_transformer_6400updates",
        _identity("word_decoding", "smn4lang", "sub01", "ablation_context_singleton"),
    ),
    _run(
        "smn4lang_mengzi_singleton_3s",
        "configs/SMN4Lang_3s_single_word_transformer.yaml",
        "outputs/SMN4Lang/word_decoding_all_words_mengzi_3s_single_word_transformer_6400updates",
        _identity("word_decoding", "smn4lang", "sub01", "ablation_window_3s_singleton"),
    ),
    _run(
        "smn4lang_mengzi_context_3s",
        "configs/SMN4Lang.yaml",
        "outputs/SMN4Lang/word_decoding_all_words_sentence_groups_6400updates",
        _identity("word_decoding", "smn4lang", "sub01", "ablation_window_3s_context"),
    ),
    _run(
        "smn4lang_mengzi_word_3s",
        "configs/SMN4Lang_conv_only.yaml",
        "outputs/SMN4Lang/word_decoding_all_words_sentence_groups_conv_only_6400updates",
        _identity("word_decoding", "smn4lang", "sub01", "ablation_window_3s_word"),
    ),
    _run(
        "smn4lang_unknown_sentence_groups",
        None,
        "outputs/SMN4Lang/word_decoding_sentence_groups_6400updates",
    ),
)


def _file_record(path, project_root):
    return {
        "relative_path": path.relative_to(project_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _manifest_digest(manifest):
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_legacy_results_manifest(project_root=PROJECT_ROOT):
    """扫描冻结的 32 个历史运行目录并计算所有科学文件摘要。"""
    project_root = Path(project_root)
    runs = []
    counts = {"runs": 0, "checkpoints": 0, "evaluations": 0, "summaries": 0, "audits": 0}
    for specification in LEGACY_RUNS:
        output_dir = project_root / specification["legacy_output"]
        if not output_dir.is_dir():
            raise FileNotFoundError(f"历史运行目录不存在：{output_dir}")
        grouped = {
            "checkpoint_files": [],
            "evaluation_files": [],
            "summary_files": [],
            "audit_files": [],
        }
        for path in sorted(
            (item for item in output_dir.rglob("*") if item.is_file()),
            key=lambda item: item.as_posix(),
        ):
            if path.name in {"best.pt", "last.pt"}:
                key = "checkpoint_files"
            elif path.name in {"training_summary.json", "diagnostic_summary.json"}:
                key = "summary_files"
            elif path.name.startswith("evaluation_") and path.suffix == ".json":
                key = "evaluation_files"
            else:
                key = "audit_files"
            grouped[key].append(_file_record(path, project_root))
        record = dict(specification)
        record.update(grouped)
        runs.append(record)
        counts["runs"] += 1
        counts["checkpoints"] += len(grouped["checkpoint_files"])
        counts["evaluations"] += len(grouped["evaluation_files"])
        counts["summaries"] += len(grouped["summary_files"])
        counts["audits"] += len(grouped["audit_files"])
    manifest = {
        "schema_version": 1,
        "status": "immutable_legacy_numeric_baseline",
        "generator": "experiments/freeze_legacy_results.py",
        "generator_sha256": file_sha256(
            PROJECT_ROOT / "experiments" / "freeze_legacy_results.py"
        ),
        "hash_contract": "sha256(canonical_json_without_manifest_sha256)",
        "run_count": len(runs),
        "counts": counts,
        "runs": runs,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    return manifest


def main():
    manifest = build_legacy_results_manifest()
    path = PROJECT_ROOT / "experiments" / "legacy_results_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"path": str(path), **manifest["counts"], "manifest_sha256": manifest["manifest_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
