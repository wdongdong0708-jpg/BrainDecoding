"""Exploratory story-specific lexical information audit (candidate_v0).

This module reads frozen lexical/event assets only.  It never reads neural arrays,
checkpoints, predictions, or model metrics, and it writes only below ``reports/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from braindecoding.config import PROJECT_ROOT
from braindecoding.data import chineseeeg2, pallier2025
from braindecoding.protocols import canonical_words

from .ovmi import load_reference_distribution


AUDIT_NAME = "story_specific_lexical_audit"
FORMULA_STATUS = "candidate_v0_not_frozen"
SMOOTHING_ALPHAS = (0.5, 1.0)
THRESHOLDS = (1, 2, 5, 10, 20)
TOP_SIZES = (20, 50, 100, 150)
REPORT_ROOT = PROJECT_ROOT / "reports" / AUDIT_NAME

DATASETS = {
    "chineseeeg2_littleprince": {
        "dataset": "ChineseEEG2 Little Prince",
        "language": "zh",
        "event_table": PROJECT_ROOT
        / "derived"
        / "chineseeeg2_littleprince"
        / "events"
        / "events.csv",
        "manifest_dir": PROJECT_ROOT / "experiments" / "manifests" / "chineseeeg2",
        "normalization": "existing_event_table_standard_word_strip_only",
        "frequency_unit": "voice_material_word_occurrence",
        "deduplication_key": ["voice_version", "source_word_id"],
        "output_name": "chineseeeg2_littleprince",
    },
    "pallier2025": {
        "dataset": "Pallier2025 Little Prince",
        "language": "fr",
        "event_table": PROJECT_ROOT
        / "derived"
        / "pallier2025"
        / "events"
        / "events.csv",
        "manifest_dir": PROJECT_ROOT / "experiments" / "manifests" / "pallier2025",
        "normalization": "strip_lower_preserve_french_accents_and_tokenization",
        "frequency_unit": "material_word_occurrence",
        "deduplication_key": ["运行编号", "BIDS事件行号"],
        "output_name": "pallier2025",
    },
}


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without interpreting the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dataset_output_directory(dataset: str) -> Path:
    """Resolve the fixed reports-only output directory for a supported dataset."""
    if dataset not in DATASETS:
        raise ValueError(f"不支持的数据集：{dataset}")
    path = REPORT_ROOT / DATASETS[dataset]["output_name"] / "candidate_v0"
    if REPORT_ROOT.resolve() not in path.resolve().parents:
        raise ValueError("故事特异词汇审计输出必须位于 reports/。")
    return path


def normalize_chinese_background_word(value) -> str:
    """Match the frozen 标准词 surface form without segmentation or conversion."""
    return str(value).strip()


def normalize_french_background_word(value) -> str:
    """Strip and lowercase while retaining accents and source token boundaries."""
    return pallier2025.normalize_french_word(value)


def _word_counts(words) -> dict[str, int]:
    counts = Counter(str(word) for word in words if str(word))
    return {word: int(counts[word]) for word in sorted(counts)}


def _ranked_words(counts: Mapping[str, int | float]) -> list[str]:
    return sorted(counts, key=lambda word: (-float(counts[word]), str(word)))


def _rank_map(counts: Mapping[str, int | float]) -> dict[str, int]:
    return {word: index for index, word in enumerate(_ranked_words(counts), start=1)}


def _chinese_material_views(table):
    """Reuse the frozen vocabulary/reference material and voice deduplication rule."""
    train = canonical_words(
        table,
        deduplicate_by=("voice_version", "source_word_id"),
        order_by=("voice_version", "chapter", "chapter_word_index", "subject_id"),
        split="train",
        trainable_only=True,
    )
    full = canonical_words(
        table,
        deduplicate_by=("voice_version", "source_word_id"),
        order_by=("voice_version", "chapter", "chapter_word_index", "subject_id"),
    )
    train = train[train["_protocol_word"].ne("")].reset_index(drop=True)
    full = full[full["_protocol_word"].ne("")].reset_index(drop=True)
    return train, full


def _pallier_material_views(table):
    """Reuse Pallier's audited material view, including the run-03 source rule."""
    full = pallier2025.material_word_view(table)
    full = full[full["_protocol_word"].ne("")].reset_index(drop=True)
    train = full[full["数据划分"].eq("train")].reset_index(drop=True)
    return train, full


def load_story_counts(dataset: str) -> dict:
    """Load train/full material-level counts and verify the frozen references."""
    if dataset not in DATASETS:
        raise ValueError(f"不支持的数据集：{dataset}")
    spec = DATASETS[dataset]
    event_path = Path(spec["event_table"])
    story_path = Path(spec["manifest_dir"]) / "story_reference.json"
    if dataset == "chineseeeg2_littleprince":
        table = chineseeeg2.载入事件表(event_path, trainable_only=False)
        train, full = _chinese_material_views(table)
        train_units = sorted(
            {
                f"ChineseEEG2|littleprince|chapter-{int(value):02d}"
                for value in train["chapter"]
            }
        )
        full_units = sorted(
            {
                f"ChineseEEG2|littleprince|chapter-{int(chapter):02d}|voice-{voice}"
                for chapter, voice in zip(full["chapter"], full["voice_version"])
            }
        )
        material_source = {
            "source_subject_rule": "stable_first_after_material_key_sort",
            "subject_repetitions_counted": False,
        }
    else:
        table = pallier2025.load_event_table(event_path, trainable_only=False)
        train, full = _pallier_material_views(table)
        train_units = sorted(train["划分单元"].astype(str).unique().tolist())
        full_units = sorted(full["运行编号"].astype(str).unique().tolist())
        source_by_run = {
            str(run): str(source)
            for run, source in full.groupby("运行编号", sort=True)[
                "_material_source_subject"
            ].first().items()
        }
        material_source = {
            "source_subject_by_run": source_by_run,
            "run_03_source_excludes": "sub-09/run-03",
            "subject_repetitions_counted": False,
        }

    train_counts = _word_counts(train["_protocol_word"])
    full_counts = _word_counts(full["_protocol_word"])
    frozen_story = {str(word): int(count) for word, count in _json(story_path).items()}
    if full_counts != frozen_story:
        raise ValueError(
            f"{dataset} material view 与冻结 story_reference.json 不一致，停止审计。"
        )

    canonical_assets = {}
    for size in TOP_SIZES:
        path = Path(spec["manifest_dir"]) / f"vocabulary_N{size}.json"
        manifest = _json(path)
        derived = _ranked_words(train_counts)[:size]
        if derived != list(manifest["vocabulary"]):
            raise ValueError(f"{dataset} train material 与冻结 N{size} 词表不一致。")
        canonical_assets[f"vocabulary_N{size}"] = {
            "path": _project_relative(path),
            "sha256": file_sha256(path),
        }

    return {
        "train_counts": train_counts,
        "full_counts": full_counts,
        "train_units": train_units,
        "full_units": full_units,
        "material_source": material_source,
        "event_table": {
            "path": _project_relative(event_path),
            "sha256": file_sha256(event_path),
        },
        "story_reference": {
            "path": _project_relative(story_path),
            "sha256": file_sha256(story_path),
            "verified_equal_to_full_story_counts": True,
        },
        "canonical_assets": canonical_assets,
    }


def _metadata_candidates(background_path: Path) -> list[Path]:
    return [
        Path(str(background_path) + ".provenance.json"),
        background_path.with_suffix(".provenance.json"),
        background_path.with_suffix(".metadata.json"),
    ]


def load_background_reference(
    path,
    *,
    normalize: Callable,
    language: str,
    metadata_path=None,
    source_label: str | None = None,
) -> tuple[dict[str, float], dict]:
    """Load through OVMI's existing reader, then apply dataset-compatible forms."""
    background_path = Path(path).resolve()
    raw = load_reference_distribution(background_path)
    normalized: dict[str, float] = {}
    for raw_word, count in raw.items():
        word = str(normalize(raw_word)).strip()
        if not word:
            raise ValueError("background normalization 产生了空词。")
        normalized[word] = normalized.get(word, 0.0) + float(count)
    if not normalized or not math.isfinite(sum(normalized.values())):
        raise ValueError("normalized background reference 无有效质量。")
    if sum(normalized.values()) <= 0:
        raise ValueError("normalized background reference 总质量必须大于零。")

    selected_metadata_path = Path(metadata_path).resolve() if metadata_path else None
    if selected_metadata_path is None:
        selected_metadata_path = next(
            (candidate for candidate in _metadata_candidates(background_path) if candidate.is_file()),
            None,
        )
    metadata = {}
    if selected_metadata_path is not None:
        metadata = _json(selected_metadata_path)
        if not isinstance(metadata, Mapping):
            raise ValueError("background metadata 必须是 JSON object。")
    label = source_label or metadata.get("source_label") or metadata.get("source")
    missing_metadata = [
        field
        for field, value in (
            ("source_label", label),
            ("version", metadata.get("version")),
            ("license", metadata.get("license")),
            ("language", metadata.get("language")),
            ("tokenization", metadata.get("tokenization")),
        )
        if value in (None, "")
    ]
    language_value = metadata.get("language")
    language_mismatch = bool(language_value and str(language_value) != language)
    return normalized, {
        "status": "available",
        "path": str(background_path),
        "sha256": file_sha256(background_path),
        "language": language,
        "metadata_language": language_value,
        "language_mismatch": language_mismatch,
        "source_label": label,
        "metadata_path": str(selected_metadata_path) if selected_metadata_path else None,
        "metadata_sha256": (
            file_sha256(selected_metadata_path) if selected_metadata_path else None
        ),
        "source_metadata": dict(metadata),
        "missing_metadata": missing_metadata,
        "raw_total_count": float(sum(raw.values())),
        "raw_type_count": int(len(raw)),
        "total_count": float(sum(normalized.values())),
        "type_count": int(len(normalized)),
        "normalization_collapsed_type_count": int(len(raw) - len(normalized)),
        "reader": "braindecoding.evaluation.ovmi.load_reference_distribution",
    }


def coverage_audit(
    story_counts: Mapping[str, int], background_counts: Mapping[str, float]
) -> dict:
    """Report token/type coverage without imposing an automatic pass threshold."""
    total_tokens = int(sum(story_counts.values()))
    story_types = set(story_counts)
    background_types = set(background_counts)
    matched_types = story_types & background_types
    matched_tokens = int(sum(story_counts[word] for word in matched_types))
    missing = sorted(
        story_types - background_types,
        key=lambda word: (-int(story_counts[word]), word),
    )
    top100 = _ranked_words(story_counts)[:100]
    top100_missing = [word for word in top100 if word not in background_types]
    return {
        "matched_token_count": matched_tokens,
        "total_token_count": total_tokens,
        "matched_token_fraction": matched_tokens / total_tokens,
        "matched_type_count": int(len(matched_types)),
        "total_type_count": int(len(story_types)),
        "matched_type_fraction": len(matched_types) / len(story_types),
        "background_missing_types": missing,
        "top100_story_frequency_background_missing": top100_missing,
        "coverage_pass_threshold": None,
        "coverage_interpretation": "human_review_required_no_automatic_pass_gate",
    }


def diagnostic_flags(word: str, background_missing: bool, train_count: int) -> dict:
    """Mark suspicious surface forms; flags never filter or rewrite a word."""
    text = str(word)
    stripped = text.strip()
    punctuation_only = bool(stripped) and all(
        unicodedata.category(char)[0] in {"P", "S"} for char in stripped
    )
    contains_digit = any(char.isdigit() for char in stripped)
    number_like = bool(stripped) and all(
        char.isdigit() or char in ".,，。+-−–—/%‰" for char in stripped
    )
    return {
        "background_missing": bool(background_missing),
        "number_like": bool(number_like),
        "punctuation_only": bool(punctuation_only),
        "contains_digit": bool(contains_digit),
        "leading_hyphen": stripped.startswith(("-", "‐", "‑", "‒", "–", "—")),
        "single_character_or_token": len(stripped) == 1,
        "very_low_train_count": int(train_count) < 2,
    }


def score_story_distribution(
    story_counts: Mapping[str, int],
    background_counts: Mapping[str, float],
    *,
    alpha: float,
    train_counts: Mapping[str, int] | None = None,
) -> dict:
    """Apply the explicitly smoothed candidate_v0 formula to observed story words."""
    if not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("smoothing alpha 必须是有限正数。")
    story = {str(word): int(count) for word, count in story_counts.items() if int(count) > 0}
    background = {
        str(word): float(count) for word, count in background_counts.items()
    }
    if not story or sum(story.values()) <= 0:
        raise ValueError("story counts 必须包含正质量。")
    if not background or not math.isfinite(sum(background.values())) or sum(background.values()) <= 0:
        raise ValueError("background counts 必须包含有限正质量。")
    if any(not math.isfinite(count) or count < 0 for count in background.values()):
        raise ValueError("background count 必须有限且非负。")

    train = {str(word): int(count) for word, count in (train_counts or story).items()}
    standard_ranks = _rank_map(train)
    total_story = float(sum(story.values()))
    total_background = float(sum(background.values()))
    universe = set(story) | set(background)
    denominator = total_background + float(alpha) * len(universe)
    rows = []
    for word, story_count in story.items():
        background_count = float(background.get(word, 0.0))
        p_story = story_count / total_story
        p_background = (background_count + float(alpha)) / denominator
        raw_log_lift = math.log2(p_story / p_background)
        signed = p_story * raw_log_lift
        positive = p_story * max(raw_log_lift, 0.0)
        flags = diagnostic_flags(word, word not in background, train.get(word, 0))
        row = {
            "word": word,
            "story_count": int(story_count),
            "train_count": int(train.get(word, 0)),
            "p_story": float(p_story),
            "background_count": background_count,
            "p_background": float(p_background),
            "raw_log_lift_bits": float(raw_log_lift),
            "signed_kl_contribution_bits": float(signed),
            "positive_story_specific_score_bits": float(positive),
            "standard_frequency_rank": standard_ranks.get(word),
            **flags,
        }
        row["diagnostic_flags"] = ";".join(
            name for name, enabled in flags.items() if enabled
        )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -row["positive_story_specific_score_bits"],
            -row["story_count"],
            row["word"],
        )
    )
    positive_sum = math.fsum(row["positive_story_specific_score_bits"] for row in rows)
    for rank, row in enumerate(rows, start=1):
        row["story_specific_rank"] = rank
        row["q_story_specific"] = (
            row["positive_story_specific_score_bits"] / positive_sum
            if positive_sum > 0
            else 0.0
        )
    total_kl = math.fsum(row["signed_kl_contribution_bits"] for row in rows)
    recomputed = math.fsum(
        row["p_story"] * math.log2(row["p_story"] / row["p_background"])
        for row in rows
    )
    if not math.isclose(total_kl, recomputed, rel_tol=1e-12, abs_tol=1e-12):
        raise ArithmeticError("signed KL contribution sum sanity check failed。")
    entropy = -math.fsum(
        row["q_story_specific"] * math.log2(row["q_story_specific"])
        for row in rows
        if row["q_story_specific"] > 0
    )
    concentration = {
        f"top{size}_cumulative_mass": float(
            math.fsum(row["q_story_specific"] for row in rows[:size])
        )
        for size in TOP_SIZES
    }
    return {
        "alpha": float(alpha),
        "universe_type_count": int(len(universe)),
        "story_total_tokens": int(total_story),
        "background_total_count": total_background,
        "rows": rows,
        "total_kl_bits": float(total_kl),
        "signed_contribution_sum_bits": float(recomputed),
        "signed_kl_sum_sanity_check": True,
        "sum_positive_story_specific_score_bits": float(positive_sum),
        "q_story_specific_entropy_bits": float(entropy),
        **concentration,
    }


def _rank_correlation(words_a: list[str], words_b: list[str]) -> float | None:
    ranks_a = {word: index for index, word in enumerate(words_a, start=1)}
    ranks_b = {word: index for index, word in enumerate(words_b, start=1)}
    common = sorted(set(words_a) & set(words_b))
    if len(common) < 2:
        return None
    left = np.asarray([ranks_a[word] for word in common], dtype=np.float64)
    right = np.asarray([ranks_b[word] for word in common], dtype=np.float64)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def ranking_comparison(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Compare deterministic top-N sets and their within-overlap rank order."""
    words_a = [row["word"] for row in rows_a]
    words_b = [row["word"] for row in rows_b]
    result = {}
    for size in TOP_SIZES:
        top_a = words_a[:size]
        top_b = words_b[:size]
        set_a, set_b = set(top_a), set(top_b)
        overlap = len(set_a & set_b)
        union = len(set_a | set_b)
        result[f"top{size}"] = {
            "set_overlap": int(overlap),
            "jaccard_similarity": float(overlap / union) if union else 1.0,
            "rank_correlation_common_items": _rank_correlation(top_a, top_b),
            "common_item_count_for_rank_correlation": int(overlap),
        }
    top50_jaccard = result["top50"]["jaccard_similarity"]
    result["ranking_sensitive_to_background_smoothing"] = top50_jaccard < 0.8
    result["sensitivity_flag_rule"] = (
        "exploratory heuristic: Top50 Jaccard < 0.8; not a confirmatory gate"
    )
    result["all_type_rank_correlation"] = _rank_correlation(words_a, words_b)
    return result


def threshold_sensitivity(rows: list[dict]) -> dict:
    """Filter only by train material count, without recomputing story scores."""
    thresholds = {}
    top50_by_threshold = {}
    for threshold in THRESHOLDS:
        eligible = [row for row in rows if row["train_count"] >= threshold]
        words = [row["word"] for row in eligible]
        top50 = words[:50]
        top50_by_threshold[threshold] = top50
        thresholds[str(threshold)] = {
            "eligible_type_count": int(len(eligible)),
            "constructible": {
                f"top{size}": len(eligible) >= size for size in TOP_SIZES
            },
            "top50_words": top50,
            "selection_count_field": "train_count",
        }
    comparisons = []
    for left_index, left in enumerate(THRESHOLDS):
        for right in THRESHOLDS[left_index + 1 :]:
            set_left = set(top50_by_threshold[left])
            set_right = set(top50_by_threshold[right])
            overlap = len(set_left & set_right)
            union = len(set_left | set_right)
            comparisons.append(
                {
                    "threshold_a": left,
                    "threshold_b": right,
                    "top50_overlap": int(overlap),
                    "top50_jaccard": float(overlap / union) if union else 1.0,
                }
            )
    return {"thresholds": thresholds, "pairwise_top50": comparisons}


def threshold_eligibility(train_counts: Mapping[str, int]) -> dict:
    """Count train-only eligible types even when no background score is available."""
    return {
        str(threshold): {
            "eligible_type_count": int(
                sum(int(count) >= threshold for count in train_counts.values())
            ),
            "constructible": {
                f"top{size}": sum(
                    int(count) >= threshold for count in train_counts.values()
                )
                >= size
                for size in TOP_SIZES
            },
            "top50_words": None,
            "top50_overlap": None,
            "reason_ranking_unavailable": "background_reference_not_provided",
            "selection_count_field": "train_count",
        }
        for threshold in THRESHOLDS
    }


def lexical_surface_diagnostics(
    full_counts: Mapping[str, int], train_counts: Mapping[str, int]
) -> dict:
    """Summarize non-destructive surface flags independently of background OOV."""
    rows = []
    flag_names = (
        "number_like",
        "punctuation_only",
        "contains_digit",
        "leading_hyphen",
        "single_character_or_token",
        "very_low_train_count",
    )
    totals = {name: {"type_count": 0, "token_count": 0} for name in flag_names}
    for word in _ranked_words(full_counts):
        flags = diagnostic_flags(word, False, train_counts.get(word, 0))
        active = [name for name in flag_names if flags[name]]
        for name in active:
            totals[name]["type_count"] += 1
            totals[name]["token_count"] += int(full_counts[word])
        if any(flags[name] for name in flag_names[:4]):
            rows.append(
                {
                    "word": word,
                    "full_story_count": int(full_counts[word]),
                    "train_count": int(train_counts.get(word, 0)),
                    "flags": ";".join(active),
                }
            )
    return {
        "flag_totals": totals,
        "potentially_suspicious_surface_tokens": rows,
        "single_character_examples_by_frequency": [
            word for word in _ranked_words(full_counts) if len(str(word).strip()) == 1
        ][:100],
        "automatic_deletion_performed": False,
        "background_missing_not_assessable_without_background": True,
    }


def frequency_matched_candidates(rows: list[dict], top_n: int = 50, per_word: int = 3):
    """Offer non-unique train-frequency matches with lower story-specific scores."""
    targets = rows[:top_n]
    excluded = {row["word"] for row in targets}
    candidates = [row for row in rows if row["word"] not in excluded]
    matches = []
    for target in targets:
        lower = [
            row
            for row in candidates
            if row["positive_story_specific_score_bits"]
            < target["positive_story_specific_score_bits"]
        ]
        lower.sort(
            key=lambda row: (
                abs(row["train_count"] - target["train_count"]),
                row["positive_story_specific_score_bits"],
                row["word"],
            )
        )
        for match in lower[:per_word]:
            matches.append(
                {
                    "story_specific_word": target["word"],
                    "train_count": target["train_count"],
                    "matched_ordinary_word": match["word"],
                    "matched_count": match["train_count"],
                    "count_difference": abs(
                        match["train_count"] - target["train_count"]
                    ),
                }
            )
    return matches


def _counts_rows(counts: Mapping[str, int], train_counts=None) -> list[dict]:
    ranks = _rank_map(counts)
    train_counts = train_counts or counts
    train_ranks = _rank_map(train_counts)
    return [
        {
            "word": word,
            "count": int(counts[word]),
            "frequency_rank": int(ranks[word]),
            "train_count": int(train_counts.get(word, 0)),
            "standard_frequency_rank": train_ranks.get(word),
        }
        for word in _ranked_words(counts)
    ]


def _score_csv_rows(rows: list[dict]) -> list[dict]:
    ordered = [
        "story_specific_rank",
        "word",
        "story_count",
        "train_count",
        "p_story",
        "background_count",
        "p_background",
        "raw_log_lift_bits",
        "signed_kl_contribution_bits",
        "positive_story_specific_score_bits",
        "q_story_specific",
        "standard_frequency_rank",
        "background_missing",
        "number_like",
        "punctuation_only",
        "contains_digit",
        "leading_hyphen",
        "single_character_or_token",
        "very_low_train_count",
        "diagnostic_flags",
    ]
    return [{field: row.get(field) for field in ordered} for row in rows]


def _fmt(value, digits=6) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _md(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _formula_markdown() -> str:
    return (
        "- `p_story(w) = story_count(w) / total_story_tokens`\n"
        "- `p_bg_alpha(w) = (background_count(w) + alpha) / "
        "(total_background + alpha * |U|)`，其中 `U` 是 background 与 story 词型并集\n"
        "- `raw_log_lift_bits = log2(p_story / p_bg_alpha)`\n"
        "- `signed_kl_contribution_bits = p_story * raw_log_lift_bits`\n"
        "- `positive_story_specific_score_bits = p_story * max(raw_log_lift_bits, 0)`\n\n"
        "最后一项经过正值截断，不称为 KL contribution。"
    )


def _blocked_markdown(title: str, frequency_top50: list[str]) -> str:
    return (
        f"# {title}\n\n"
        "- status: `blocked`\n"
        "- reason: `background_reference_not_provided`\n"
        "- standard frequency Top50: "
        + "、".join(f"`{_md(word)}`" for word in frequency_top50)
        + "\n\n"
        "未提供普通语言 background reference，因此未计算、伪造或排序任何 "
        "story-specific score。"
    )


def _blocked_threshold_markdown(eligibility: dict) -> str:
    lines = [
        "# Minimum train material occurrence sensitivity",
        "",
        "以下 eligible type count 只来自 train material counts。由于没有 background，"
        "Top50 story-specific ranking、threshold 间 overlap/Jaccard 均未计算。",
        "",
        "| min train count | eligible types | Top20 | Top50 | Top100 | Top150 | Top50 ranking |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for threshold in THRESHOLDS:
        row = eligibility[str(threshold)]
        flags = row["constructible"]
        lines.append(
            f"| {threshold} | {row['eligible_type_count']} | {flags['top20']} | "
            f"{flags['top50']} | {flags['top100']} | {flags['top150']} | "
            "blocked: background_reference_not_provided |"
        )
    return "\n".join(lines)


def _blocked_surface_markdown(diagnostics: dict) -> str:
    rows = diagnostics["potentially_suspicious_surface_tokens"]
    lines = [
        "# High-score background-missing words",
        "",
        "status: `blocked`; reason: `background_reference_not_provided`。因此无法判断 "
        "background-missing，更不能定义 high-score OOV。下面仅列与 background 无关的"
        "表面 token flags；没有自动删除。",
        "",
        "| word | full story count | train count | surface flags |",
        "|---|---:|---:|---|",
    ]
    for row in rows[:100]:
        lines.append(
            f"| {_md(row['word'])} | {row['full_story_count']} | "
            f"{row['train_count']} | {row['flags']} |"
        )
    if not rows:
        lines.append("| — | — | — | no number/punctuation/digit/leading-hyphen tokens |")
    return "\n".join(lines)


def _coverage_table(coverage: dict) -> str:
    return (
        "| coverage | matched | total | fraction |\n"
        "|---|---:|---:|---:|\n"
        f"| token mass | {coverage['matched_token_count']} | "
        f"{coverage['total_token_count']} | {coverage['matched_token_fraction']:.6f} |\n"
        f"| types | {coverage['matched_type_count']} | "
        f"{coverage['total_type_count']} | {coverage['matched_type_fraction']:.6f} |"
    )


def _frequency_comparison_markdown(
    frequency_words: list[str], scored: dict[float, dict]
) -> tuple[str, str]:
    top100_lines = [
        "# Standard frequency 与 story-specific lexical information",
        "",
        "标准频率 rank 来自严格 train-only material counts。下面的 story-specific "
        "rank 也只使用 train story distribution；full-story score 不参与候选排序。",
    ]
    direct_lines = [
        "# Standard frequency Top50 vs story-specific Top50",
        "",
    ]
    frequency_set = set(frequency_words[:50])
    for alpha in SMOOTHING_ALPHAS:
        rows = scored[alpha]["rows"]
        top100_lines.extend(
            [
                "",
                f"## alpha = {alpha}",
                "",
                "| rank | word | train count | story p | bg p | log-lift | "
                "story-specific score | standard frequency rank |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows[:100]:
            top100_lines.append(
                f"| {row['story_specific_rank']} | {_md(row['word'])} | "
                f"{row['train_count']} | {row['p_story']:.8g} | "
                f"{row['p_background']:.8g} | {row['raw_log_lift_bits']:.6f} | "
                f"{row['positive_story_specific_score_bits']:.8g} | "
                f"{row['standard_frequency_rank'] or '—'} |"
            )
        story_words = [row["word"] for row in rows[:50]]
        story_set = set(story_words)
        direct_lines.extend(
            [
                f"## alpha = {alpha}",
                "",
                f"- Top50 overlap: `{len(frequency_set & story_set)}`",
                "- 仅在 frequency Top50: "
                + "、".join(
                    f"`{_md(word)}`" for word in frequency_words[:50] if word not in story_set
                ),
                "- 仅在 story-specific Top50: "
                + "、".join(
                    f"`{_md(word)}`" for word in story_words if word not in frequency_set
                ),
                "",
            ]
        )
    return "\n".join(top100_lines), "\n".join(direct_lines)


def _smoothing_markdown(comparisons: dict) -> str:
    lines = [
        "# Smoothing sensitivity",
        "",
        "比较 half-count (`alpha=0.5`) 与 one-count (`alpha=1.0`)。rank "
        "correlation 是两套 Top-N 交集词的名次 Pearson correlation（即无并列的 "
        "Spearman correlation）。",
    ]
    for scope in ("train", "full_story"):
        block = comparisons[scope]
        lines.extend(
            [
                "",
                f"## {scope}",
                "",
                "| N | set overlap | Jaccard | rank correlation | common ranks |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for size in TOP_SIZES:
            row = block[f"top{size}"]
            lines.append(
                f"| {size} | {row['set_overlap']} | "
                f"{row['jaccard_similarity']:.6f} | "
                f"{_fmt(row['rank_correlation_common_items'])} | "
                f"{row['common_item_count_for_rank_correlation']} |"
            )
        lines.extend(
            [
                "",
                "- all-type rank correlation: "
                f"`{_fmt(block['all_type_rank_correlation'])}`",
                "- ranking_sensitive_to_background_smoothing: "
                f"`{str(block['ranking_sensitive_to_background_smoothing']).lower()}`",
                f"- flag rule: {block['sensitivity_flag_rule']}",
            ]
        )
    return "\n".join(lines)


def _threshold_markdown(thresholds_by_alpha: dict) -> str:
    lines = [
        "# Minimum train material occurrence sensitivity",
        "",
        "阈值只读取 `train_count`。score 在筛选前由完整 train distribution 计算，"
        "没有读取 val/test support，也没有冻结阈值。",
    ]
    for alpha in SMOOTHING_ALPHAS:
        audit = thresholds_by_alpha[alpha]
        lines.extend(
            [
                "",
                f"## alpha = {alpha}",
                "",
                "| min train count | eligible types | Top20 | Top50 | Top100 | Top150 | Top50 words |",
                "|---:|---:|---|---|---|---|---|",
            ]
        )
        for threshold in THRESHOLDS:
            row = audit["thresholds"][str(threshold)]
            flags = row["constructible"]
            words = "、".join(_md(word) for word in row["top50_words"])
            lines.append(
                f"| {threshold} | {row['eligible_type_count']} | "
                f"{flags['top20']} | {flags['top50']} | {flags['top100']} | "
                f"{flags['top150']} | {words} |"
            )
        lines.extend(
            [
                "",
                "| threshold A | threshold B | Top50 overlap | Jaccard |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in audit["pairwise_top50"]:
            lines.append(
                f"| {row['threshold_a']} | {row['threshold_b']} | "
                f"{row['top50_overlap']} | {row['top50_jaccard']:.6f} |"
            )
    return "\n".join(lines)


def _high_missing_markdown(scored_by_scope: dict) -> str:
    lines = [
        "# High-score background-missing words",
        "",
        "这些词在 background 中原始 count 为 0；表中分数完全依赖已显式记录的 "
        "smoothing。它们不会被自动删除。",
    ]
    for scope in ("train", "full_story"):
        for alpha in SMOOTHING_ALPHAS:
            rows = scored_by_scope[scope][alpha]["rows"]
            missing = [row for row in rows if row["background_missing"]]
            top50_missing = sum(row["background_missing"] for row in rows[:50])
            lines.extend(
                [
                    "",
                    f"## {scope}, alpha = {alpha}",
                    "",
                    f"- Top50 background-missing count: `{top50_missing}`",
                    "- background_reference_insufficient_for_story_specific_ranking: "
                    f"`{str(top50_missing >= 25).lower()}`",
                    "- exploratory flag rule: at least half of Top50 is background-missing",
                    "",
                    "| rank | word | story count | train count | score | flags |",
                    "|---:|---|---:|---:|---:|---|",
                ]
            )
            for row in missing[:100]:
                lines.append(
                    f"| {row['story_specific_rank']} | {_md(row['word'])} | "
                    f"{row['story_count']} | {row['train_count']} | "
                    f"{row['positive_story_specific_score_bits']:.8g} | "
                    f"{row['diagnostic_flags']} |"
                )
    return "\n".join(lines)


def _frequency_matches_markdown(matches_by_alpha: dict) -> str:
    lines = [
        "# Exploratory frequency-matched candidate diagnostics",
        "",
        "每个高 story-specific 词列出最多三个 train count 最接近、且 score "
        "更低的普通词候选。该表不是唯一 matched set，未用于模型评价。",
    ]
    for alpha in SMOOTHING_ALPHAS:
        lines.extend(
            [
                "",
                f"## alpha = {alpha}",
                "",
                "| story-specific word | train count | matched ordinary word | matched count | count difference |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for row in matches_by_alpha[alpha]:
            lines.append(
                f"| {_md(row['story_specific_word'])} | {row['train_count']} | "
                f"{_md(row['matched_ordinary_word'])} | {row['matched_count']} | "
                f"{row['count_difference']} |"
            )
    return "\n".join(lines)


def _normalization_markdown(dataset: str, material: dict) -> str:
    spec = DATASETS[dataset]
    if dataset == "chineseeeg2_littleprince":
        details = (
            "故事侧直接使用 frozen event table 的 `标准词/normalized_word`。background "
            "只做 `strip`；不重新分词、不做简繁转换、词干化、同义词或概念合并。"
        )
    else:
        details = (
            "故事与 background 均只做 `strip` + `lower`；保留法语重音和 source "
            "tokenization。不合并 `j + avais`，不 lemmatize、accent-strip、stem 或 "
            "clitic-merge。"
        )
    return (
        "# Normalization audit\n\n"
        f"- dataset: `{spec['dataset']}`\n"
        f"- language: `{spec['language']}`\n"
        f"- normalization: `{spec['normalization']}`\n"
        f"- full story equals frozen story_reference: "
        f"`{str(material['story_reference']['verified_equal_to_full_story_counts']).lower()}`\n\n"
        f"{details}\n\n"
        "异常 token 只加 diagnostic flags，不会自动删除。"
    )


def _background_requirements_markdown(dataset: str, background: dict) -> str:
    spec = DATASETS[dataset]
    lines = [
        "# Background reference requirements",
        "",
        f"- dataset language: `{spec['language']}`",
        f"- status: `{background['status']}`",
        f"- path: `{background.get('path') or 'not provided'}`",
        f"- SHA-256: `{background.get('sha256') or 'not available'}`",
        f"- source label: `{background.get('source_label') or 'missing_metadata'}`",
        "- accepted input: JSON `{word: count}` or CSV `word,count`",
        "- validation: non-empty; finite nonnegative counts; positive total mass",
        "- reader: `braindecoding.evaluation.ovmi.load_reference_distribution`",
        "",
        "不会自动下载、选择或将 Wikipedia/OpenSubtitles/SUBTLEX/随机 corpus "
        "指定为正式 background。",
    ]
    if background.get("missing_metadata"):
        lines.extend(
            [
                "",
                "missing_metadata: " + ", ".join(background["missing_metadata"]),
            ]
        )
    return "\n".join(lines)


def _summary_markdown(manifest: dict) -> str:
    counts = manifest["counting"]
    lines = [
        f"# {manifest['dataset']} 故事特异词汇审计 v0",
        "",
        "This is an exploratory lexical audit. No model or test evaluation was performed.",
        "",
        "本审计衡量相对于普通语言分布异常常见的词，即 story-specific lexical "
        "information；它不表示 semantic importance、剧情因果或人物重要度。",
        "",
        "## 1. 数据集与材料计数",
        "",
        f"- dataset: `{manifest['dataset']}`",
        f"- language: `{manifest['language']}`",
        f"- frequency_unit: `{counts['frequency_unit']}`",
        f"- deduplication_key: `{counts['deduplication_key']}`",
        f"- train story: `{counts['train_token_count']}` tokens / "
        f"`{counts['train_type_count']}` types",
        f"- full story: `{counts['full_story_token_count']}` tokens / "
        f"`{counts['full_story_type_count']}` types",
        "- full story counts verified exactly equal to frozen `story_reference.json`.",
        "",
        "## 2. Background source 与 coverage",
        "",
        f"- status: `{manifest['status']}`",
        f"- background path: `{manifest['background'].get('path') or 'not provided'}`",
        f"- source label: `{manifest['background'].get('source_label') or 'missing_metadata'}`",
    ]
    if manifest["status"] == "blocked":
        lines.extend(
            [
                f"- reason: `{manifest['reason']}`",
                "- coverage: `not computed`",
                "- tokenization_mismatch_risk: `not assessable without background`",
                "",
                "仓库、`artifacts/`、本地 reference 配置及已知本地词频文件检索未找到"
                "可用的中文或法语 background。按照审计合同，本数据集不计算 "
                "story-specific score。",
            ]
        )
    else:
        lines.extend(
            [
                f"- SHA-256: `{manifest['background']['sha256']}`",
                f"- background total/type count: "
                f"`{manifest['background']['total_count']}` / "
                f"`{manifest['background']['type_count']}`",
                "",
                "### Train coverage",
                "",
                _coverage_table(manifest["coverage"]["train"]),
                "",
                "### Full-story coverage",
                "",
                _coverage_table(manifest["coverage"]["full_story"]),
                "",
                "Top-100 story-frequency missing words: "
                + "、".join(
                    f"`{_md(word)}`"
                    for word in manifest["coverage"]["full_story"][
                        "top100_story_frequency_background_missing"
                    ]
                ),
                "",
                "coverage 没有自动通过阈值；可信度由人根据 token/type coverage、"
                "missing types 和 metadata 共同审阅。",
            ]
        )
    lines.extend(["", "## 3. Candidate v0 formula", "", _formula_markdown()])
    if manifest["status"] == "completed":
        lines.extend(["", "## 4. Smoothing stability", ""])
        train_smoothing = manifest["smoothing_sensitivity"]["train"]
        for size in TOP_SIZES:
            row = train_smoothing[f"top{size}"]
            lines.append(
                f"- Top{size}: overlap `{row['set_overlap']}`, Jaccard "
                f"`{row['jaccard_similarity']:.6f}`, rank correlation "
                f"`{_fmt(row['rank_correlation_common_items'])}`"
            )
        lines.extend(
            [
                "",
                "## 5. Threshold stability",
                "",
                "见 `threshold_sensitivity.md`；同时报告 1/2/5/10/20，未冻结阈值。",
                "",
                "## 6. Frequency Top50 vs story-specific Top50",
                "",
                "见 `standard_frequency_top50_vs_story_specific_top50.md`。",
                "",
                "## 7. Top story-specific words",
                "",
            ]
        )
        for alpha in SMOOTHING_ALPHAS:
            top = manifest["results"]["train"][f"alpha{str(alpha).replace('.', 'p')}"][
                "top50_words"
            ]
            lines.append(
                f"- alpha={alpha}: " + "、".join(f"`{_md(word)}`" for word in top)
            )
        lines.extend(
            [
                "",
                "## 8. Suspicious/OOV tokens",
                "",
                "见 `high_score_background_missing.md` 和 score CSV 的 diagnostic flags。",
                "",
                "## 9. 是否建议冻结",
                "",
                "`no`。candidate_v0 仍是 exploratory，必须先人工审阅 background "
                "provenance、coverage、tokenization compatibility、smoothing 与 threshold "
                "敏感性。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## 4. 未计算项目",
                "",
                "smoothing stability、threshold Top50 排名/overlap、story-specific Top50、"
                "background OOV 高分词、q concentration、signed KL 与 positive score "
                "mass 均未计算。train-only threshold eligible type count 和 surface "
                "token flags 已完成，见相应报告。",
                "",
                "## 5. Standard frequency Top50",
                "",
                "、".join(
                    f"`{_md(word)}`" for word in manifest["standard_frequency_top50"]
                ),
                "",
                "## 6. 是否建议冻结",
                "",
                "`no`。background_reference_not_provided；不得冻结候选词表。",
            ]
        )
    lines.extend(
        [
            "",
            "## Safeguards",
            "",
            "- canonical frequency candidate vocabulary：未修改",
            "- standard story reference：未修改",
            "- split / event table / derived signals or text：未修改",
            "- checkpoint / model forward / test neural arrays：未读取",
            "- test prediction / test metric：未生成、未运行、未检查",
        ]
    )
    return "\n".join(lines)


def _result_summary(scored: dict) -> dict:
    rows = scored["rows"]
    return {
        key: value
        for key, value in scored.items()
        if key != "rows"
    } | {
        "top50_words": [row["word"] for row in rows[:50]],
        "top50_background_missing_count": int(
            sum(row["background_missing"] for row in rows[:50])
        ),
        "suspicious_top100": [
            {"word": row["word"], "flags": row["diagnostic_flags"]}
            for row in rows[:100]
            if row["diagnostic_flags"]
        ],
    }


def run_dataset_audit(
    dataset: str,
    *,
    background=None,
    background_metadata=None,
    source_label: str | None = None,
) -> tuple[Path, dict]:
    """Run one reports-only lexical audit and return its manifest."""
    if dataset not in DATASETS:
        raise ValueError(f"不支持的数据集：{dataset}")
    spec = DATASETS[dataset]
    output = dataset_output_directory(dataset)
    output.mkdir(parents=True, exist_ok=True)
    material = load_story_counts(dataset)
    train_counts = material["train_counts"]
    full_counts = material["full_counts"]
    frequency_top50 = _ranked_words(train_counts)[:50]

    _write_csv(
        output / "train_story_counts.csv",
        _counts_rows(train_counts),
        ["word", "count", "frequency_rank", "train_count", "standard_frequency_rank"],
    )
    _write_csv(
        output / "full_story_counts.csv",
        _counts_rows(full_counts, train_counts),
        ["word", "count", "frequency_rank", "train_count", "standard_frequency_rank"],
    )

    background_block = {
        "status": "not_provided",
        "path": None,
        "sha256": None,
        "language": spec["language"],
        "source_label": None,
        "total_count": None,
        "type_count": None,
        "missing_metadata": [
            "source_label",
            "version",
            "license",
            "language",
            "tokenization",
        ],
        "reader": "braindecoding.evaluation.ovmi.load_reference_distribution",
    }
    manifest = {
        "audit": AUDIT_NAME,
        "dataset": spec["dataset"],
        "dataset_key": dataset,
        "language": spec["language"],
        "status": "blocked" if background is None else "running",
        "reason": "background_reference_not_provided" if background is None else None,
        "exploratory_only": True,
        "formula_status": FORMULA_STATUS,
        "event_table": material["event_table"],
        "story_reference": material["story_reference"],
        "canonical_assets_read_only": material["canonical_assets"],
        "background": background_block,
        "counting": {
            "frequency_unit": spec["frequency_unit"],
            "deduplication_key": spec["deduplication_key"],
            "normalization": spec["normalization"],
            "train_split_units": material["train_units"],
            "full_story_units": material["full_units"],
            "train_token_count": int(sum(train_counts.values())),
            "train_type_count": int(len(train_counts)),
            "full_story_token_count": int(sum(full_counts.values())),
            "full_story_type_count": int(len(full_counts)),
            "material_source": material["material_source"],
        },
        "formula": {
            "p_story": "story_count / total_story_tokens",
            "p_background": "(background_count + alpha) / (total_background + alpha * |U|)",
            "universe": "union(background vocabulary, story vocabulary)",
            "raw_log_lift": "log2(p_story / p_background)",
            "signed_kl_contribution": "p_story * raw_log_lift",
            "positive_story_specific_score": "p_story * max(raw_log_lift, 0)",
            "positive_score_is_not_named_kl_contribution": True,
        },
        "smoothing_policies": [{"alpha": alpha} for alpha in SMOOTHING_ALPHAS],
        "thresholds": list(THRESHOLDS),
        "threshold_eligibility_train_only": threshold_eligibility(train_counts),
        "lexical_surface_diagnostics": lexical_surface_diagnostics(
            full_counts, train_counts
        ),
        "standard_frequency_top50": frequency_top50,
        "candidate_ranking_source": "train_story_counts_plus_background_only",
        "full_story_role": "reference_distribution_diagnostic_only",
        "test_neural_data_opened": False,
        "test_model_evaluation": "not_run",
        "test_predictions_generated": False,
        "test_metrics_inspected": False,
        "model_training_performed": False,
        "checkpoint_opened": False,
        "canonical_vocabulary_modified": False,
        "story_reference_modified": False,
        "split_modified": False,
        "event_table_modified": False,
        "derived_signals_or_text_modified": False,
    }

    _write_text(output / "normalization_audit.md", _normalization_markdown(dataset, material))
    if background is None:
        blocked_files = {
            "smoothing_sensitivity.md": "Smoothing sensitivity",
            "frequency_vs_story_specific.md": "Frequency vs story-specific Top100",
            "standard_frequency_top50_vs_story_specific_top50.md": "Frequency Top50 vs story-specific Top50",
            "frequency_matched_candidates.md": "Frequency-matched candidates",
        }
        for filename, title in blocked_files.items():
            _write_text(output / filename, _blocked_markdown(title, frequency_top50))
        _write_text(
            output / "threshold_sensitivity.md",
            _blocked_threshold_markdown(manifest["threshold_eligibility_train_only"]),
        )
        _write_text(
            output / "high_score_background_missing.md",
            _blocked_surface_markdown(manifest["lexical_surface_diagnostics"]),
        )
        _write_text(
            output / "background_requirements.md",
            _background_requirements_markdown(dataset, background_block),
        )
        _write_text(output / "summary.md", _summary_markdown(manifest))
        _write_json(output / "exploratory_manifest.json", manifest)
        return output, manifest

    normalizer = (
        normalize_chinese_background_word
        if dataset == "chineseeeg2_littleprince"
        else normalize_french_background_word
    )
    background_counts, background_block = load_background_reference(
        background,
        normalize=normalizer,
        language=spec["language"],
        metadata_path=background_metadata,
        source_label=source_label,
    )
    manifest["background"] = background_block
    manifest["status"] = "completed"
    manifest["reason"] = None
    coverage = {
        "train": coverage_audit(train_counts, background_counts),
        "full_story": coverage_audit(full_counts, background_counts),
    }
    manifest["coverage"] = coverage
    manifest["tokenization_mismatch_risk"] = {
        "status": "human_review_required",
        "metadata_tokenization_missing": "tokenization"
        in background_block["missing_metadata"],
        "language_mismatch": background_block["language_mismatch"],
        "train_top100_missing_count": len(
            coverage["train"]["top100_story_frequency_background_missing"]
        ),
        "full_top100_missing_count": len(
            coverage["full_story"]["top100_story_frequency_background_missing"]
        ),
        "automatic_compatibility_decision": None,
    }

    scored = {"train": {}, "full_story": {}}
    for alpha in SMOOTHING_ALPHAS:
        scored["train"][alpha] = score_story_distribution(
            train_counts,
            background_counts,
            alpha=alpha,
            train_counts=train_counts,
        )
        scored["full_story"][alpha] = score_story_distribution(
            full_counts,
            background_counts,
            alpha=alpha,
            train_counts=train_counts,
        )
        alpha_name = str(alpha).replace(".", "p")
        for scope, filename in (
            ("train", f"train_word_scores_alpha{alpha_name}.csv"),
            ("full_story", f"full_story_word_scores_alpha{alpha_name}.csv"),
        ):
            rows = _score_csv_rows(scored[scope][alpha]["rows"])
            _write_csv(output / filename, rows, list(rows[0]))

    smoothing = {
        scope: ranking_comparison(scored[scope][0.5]["rows"], scored[scope][1.0]["rows"])
        for scope in ("train", "full_story")
    }
    threshold_blocks = {
        alpha: threshold_sensitivity(scored["train"][alpha]["rows"])
        for alpha in SMOOTHING_ALPHAS
    }
    matches = {
        alpha: frequency_matched_candidates(scored["train"][alpha]["rows"])
        for alpha in SMOOTHING_ALPHAS
    }
    manifest["smoothing_sensitivity"] = smoothing
    manifest["threshold_sensitivity"] = {
        f"alpha{str(alpha).replace('.', 'p')}": threshold_blocks[alpha]
        for alpha in SMOOTHING_ALPHAS
    }
    manifest["results"] = {
        scope: {
            f"alpha{str(alpha).replace('.', 'p')}": _result_summary(scored[scope][alpha])
            for alpha in SMOOTHING_ALPHAS
        }
        for scope in ("train", "full_story")
    }
    manifest["background_reference_insufficient_for_story_specific_ranking"] = {
        scope: {
            f"alpha{str(alpha).replace('.', 'p')}": (
                manifest["results"][scope][f"alpha{str(alpha).replace('.', 'p')}"][
                    "top50_background_missing_count"
                ]
                >= 25
            )
            for alpha in SMOOTHING_ALPHAS
        }
        for scope in ("train", "full_story")
    }
    manifest["background_insufficiency_flag_rule"] = (
        "exploratory warning when at least half of Top50 is background-missing; "
        "not a confirmatory gate"
    )

    reference = {
        "dataset": spec["dataset"],
        "language": spec["language"],
        "exploratory_only": True,
        "formula_status": FORMULA_STATUS,
        "candidate_selection_source": "train_story_counts_plus_background_only",
        "full_story_role": "reference_distribution_diagnostic_only",
        "distributions": {
            scope: {
                f"alpha{str(alpha).replace('.', 'p')}": {
                    row["word"]: row["q_story_specific"]
                    for row in scored[scope][alpha]["rows"]
                    if row["positive_story_specific_score_bits"] > 0
                }
                for alpha in SMOOTHING_ALPHAS
            }
            for scope in ("train", "full_story")
        },
    }
    _write_json(output / "story_specific_reference_candidate_v0.json", reference)
    frequency_md, direct_md = _frequency_comparison_markdown(frequency_top50, scored["train"])
    _write_text(output / "frequency_vs_story_specific.md", frequency_md)
    _write_text(
        output / "standard_frequency_top50_vs_story_specific_top50.md", direct_md
    )
    _write_text(output / "smoothing_sensitivity.md", _smoothing_markdown(smoothing))
    _write_text(output / "threshold_sensitivity.md", _threshold_markdown(threshold_blocks))
    _write_text(
        output / "high_score_background_missing.md", _high_missing_markdown(scored)
    )
    _write_text(
        output / "frequency_matched_candidates.md", _frequency_matches_markdown(matches)
    )
    _write_text(
        output / "background_requirements.md",
        _background_requirements_markdown(dataset, background_block),
    )
    _write_text(output / "summary.md", _summary_markdown(manifest))
    _write_json(output / "exploratory_manifest.json", manifest)
    return output, manifest


def write_cross_language_summary() -> Path:
    """Summarize only lexical distribution statistics; never align translations."""
    lines = [
        "# Chinese/French Little Prince lexical audit summary",
        "",
        "This is an exploratory lexical audit. No model or test evaluation was performed.",
        "",
        "这里只并列 material token/type count、background coverage 与 score "
        "concentration；不翻译、不做实体对齐，也不推断跨语言脑表征一致。",
        "",
        "| dataset | language | status | train tokens/types | full tokens/types | "
        "full token coverage | full type coverage |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    manifests = []
    for dataset in DATASETS:
        path = dataset_output_directory(dataset) / "exploratory_manifest.json"
        if not path.is_file():
            continue
        manifest = _json(path)
        manifests.append(manifest)
        counting = manifest["counting"]
        coverage = manifest.get("coverage", {}).get("full_story", {})
        lines.append(
            f"| {manifest['dataset']} | {manifest['language']} | {manifest['status']} | "
            f"{counting['train_token_count']}/{counting['train_type_count']} | "
            f"{counting['full_story_token_count']}/{counting['full_story_type_count']} | "
            f"{_fmt(coverage.get('matched_token_fraction'))} | "
            f"{_fmt(coverage.get('matched_type_fraction'))} |"
        )
    for manifest in manifests:
        lines.extend(["", f"## {manifest['dataset']}", ""])
        if manifest["status"] != "completed":
            lines.append(
                "`background_reference_not_provided`: score concentration、entropy、"
                "cumulative mass 与 total signed KL 未计算。"
            )
            continue
        lines.extend(
            [
                "| scope | alpha | Top20 mass | Top50 mass | Top100 mass | Top150 mass | entropy bits | total signed KL bits | positive score sum bits |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for scope in ("train", "full_story"):
            for alpha in SMOOTHING_ALPHAS:
                result = manifest["results"][scope][
                    f"alpha{str(alpha).replace('.', 'p')}"
                ]
                lines.append(
                    f"| {scope} | {alpha} | {result['top20_cumulative_mass']:.6f} | "
                    f"{result['top50_cumulative_mass']:.6f} | "
                    f"{result['top100_cumulative_mass']:.6f} | "
                    f"{result['top150_cumulative_mass']:.6f} | "
                    f"{result['q_story_specific_entropy_bits']:.6f} | "
                    f"{result['total_kl_bits']:.6f} | "
                    f"{result['sum_positive_story_specific_score_bits']:.6f} |"
                )
    path = REPORT_ROOT / "cross_language_summary.md"
    _write_text(path, "\n".join(lines))
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Story-specific lexical audit v0 (reports-only, no model evaluation)."
    )
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument(
        "--background",
        help="Explicit JSON/CSV ordinary-language frequency reference. Omit to write a blocked audit.",
    )
    parser.add_argument("--background-metadata", help="Optional JSON provenance metadata.")
    parser.add_argument("--source-label", help="Optional explicit source label.")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    output, manifest = run_dataset_audit(
        args.dataset,
        background=args.background,
        background_metadata=args.background_metadata,
        source_label=args.source_label,
    )
    cross = write_cross_language_summary()
    print(f"status={manifest['status']}")
    print(f"output={output}")
    print(f"cross_language_summary={cross}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
