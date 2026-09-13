from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from braindecoding.data import pallier2025
from braindecoding.data import build as build_derived_data
from braindecoding.events import CORE_EVENT_COLUMNS, validate_event_table


def _events(words):
    return pd.DataFrame(
        {
            "onset": [10.0 + index for index in range(len(words))],
            "duration": [0.25] * len(words),
            "trial_type": ["Word"] * len(words),
            "stimulus": words,
        }
    )


def _extra(words, *, onset_start=100.0):
    return pd.DataFrame(
        {
            "word": words,
            "onset": [onset_start + index for index in range(len(words))],
            "sequence_id": [4] * len(words),
            "is_last_word": [False] * (len(words) - 1) + [True],
            "pos": ["NC"] * len(words),
            "content_word": [True] * len(words),
            "n_closing": [1] * len(words),
            "_标准词": [pallier2025.normalize_french_word(word) for word in words],
        }
    )


def _header():
    return {"first_time_seconds": 0.0, "last_time_seconds": 1000.0}


def test_pallier_subject_discovery_and_ninety_recording_metadata(tmp_path):
    for subject in pallier2025.SUBJECTS:
        for run in pallier2025.RUNS:
            paths = pallier2025.recording_paths(tmp_path, subject, run)
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
    assert pallier2025.discover_subjects(tmp_path) == list(pallier2025.SUBJECTS)
    records = pallier2025.discover_recordings(tmp_path)
    assert len(records) == 90
    assert records[0]["subject"] == "sub-01"
    assert records[-1]["subject"] == "sub-10"


def test_pallier_config_uses_word_decoding_canonical_location():
    root = Path(__file__).resolve().parents[1]
    expected = root / "configs/word_decoding/pallier2025/base.yaml"
    assert build_derived_data._config_path("pallier2025") == expected
    assert expected.is_file()
    assert not (root / "configs/data/pallier2025.yaml").exists()


def test_frozen_run_split_is_seven_one_one_and_run_level():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/manifests/pallier2025/run_split.json"
    split = pallier2025.load_run_split(path)
    assert [sum(value == name for value in split.values()) for name in ("train", "val", "test")] == [7, 1, 1]
    assert split == {
        "run-01": "train",
        "run-02": "train",
        "run-03": "train",
        "run-04": "train",
        "run-05": "train",
        "run-08": "train",
        "run-09": "train",
        "run-07": "val",
        "run-06": "test",
    }
    payload = json.loads(path.read_text(encoding="utf-8"))
    generated, ranking = pallier2025.deterministic_run_split()
    assert generated == split
    assert ranking == payload["hash_ranking"]
    assert payload["strategy"]["uses_word_labels"] is False
    assert payload["strategy"]["uses_vocabulary_support"] is False


@pytest.mark.parametrize(
    ("source", "expected"),
    [("  Étoile  ", "étoile"), ("J'avais", "j'avais"), ("arc-en-ciel", "arc-en-ciel")],
)
def test_french_normalization_preserves_accents_and_tokenization(source, expected):
    assert pallier2025.normalize_french_word(source) == expected


def test_extra_info_blank_tokens_are_removed_before_alignment(tmp_path):
    path = tmp_path / "extra.tsv"
    path.write_text(
        "word,sequence_id,is_last_word,pos,content_word,n_closing,onset\n"
        "Le,1,False,DET,False,1,999\n"
        ",1,False,PUNCT,False,0,1000\n"
        "Prince,1,True,NC,True,2,1001\n",
        encoding="utf-8",
    )
    extra, blank_count = pallier2025._read_extra_info(path)
    assert blank_count == 1
    aligned, audit, excluded = pallier2025.align_recording_metadata(
        _events(["Le", "Prince"]),
        extra,
        subject="sub-01",
        run="run-01",
        blank_token_count=blank_count,
    )
    assert aligned["_标准词"].tolist() == ["le", "prince"]
    assert audit["edit_distance"] == 0
    assert excluded is False


def test_canonical_rows_use_bids_onset_and_fixed_chinese_columns():
    events = _events(["Étoile", "Prince"])
    extra = _extra(["Étoile", "Prince"], onset_start=900.0)
    table = pallier2025._recording_table(
        events,
        extra,
        subject="sub-01",
        run="run-07",
        split="val",
        header=_header(),
        recording_excluded=False,
        exclusion_reason="",
    )
    assert tuple(table.columns[:14]) == CORE_EVENT_COLUMNS
    assert table.columns.tolist() == list(pallier2025.PERSISTED_COLUMNS)
    assert table["开始时间"].tolist() == events["onset"].tolist()
    assert table["结束时间"].tolist() == [10.25, 11.25]
    assert "onset" not in table.columns
    assert table["上下文编号"].tolist() == [
        "pallier2025|run-07|sequence-004",
        "pallier2025|run-07|sequence-004",
    ]
    assert table["记录内序号"].tolist() == [0, 1]
    assert table["数据划分"].eq("val").all()
    validate_event_table(table)


def test_unapproved_extra_info_edit_blocks_build():
    with pytest.raises(ValueError, match="未经批准"):
        pallier2025.align_recording_metadata(
            _events(["un", "mot"]),
            _extra(["un", "autre"]),
            subject="sub-02",
            run="run-04",
            blank_token_count=0,
        )


def test_explicit_qc_mapping_preserves_bids_rows_but_excludes_recording():
    events = _events(["jour", "toujours", "le"])
    extra = _extra(["le", "jour", "toujours"])
    artifact = {
        "recording": {"subject": "sub-09", "run": "run-03"},
        "decision": "exclude_recording",
        "metadata_mapping_overrides": [
            {"bids_event_row": 2, "extra_word_index": 0}
        ],
    }
    aligned, audit, excluded = pallier2025.align_recording_metadata(
        events,
        extra,
        subject="sub-09",
        run="run-03",
        blank_token_count=0,
        qc_artifact=artifact,
    )
    assert aligned["_标准词"].tolist() == ["jour", "toujours", "le"]
    assert audit["edit_distance"] == 2
    assert excluded is True
    table = pallier2025._recording_table(
        events,
        aligned,
        subject="sub-09",
        run="run-03",
        split="train",
        header=_header(),
        recording_excluded=True,
        exclusion_reason="冻结 QC 排除",
    )
    assert table["词"].tolist() == events["stimulus"].tolist()
    assert not table["是否可训练"].any()
    assert table["排除原因"].eq("冻结 QC 排除").all()


def test_material_validation_ignores_only_fully_excluded_recording():
    first = pallier2025._recording_table(
        _events(["un", "mot"]),
        _extra(["un", "mot"]),
        subject="sub-01",
        run="run-01",
        split="train",
        header=_header(),
        recording_excluded=False,
        exclusion_reason="",
    )
    excluded = pallier2025._recording_table(
        _events(["autre", "texte"]),
        _extra(["autre", "texte"]),
        subject="sub-02",
        run="run-01",
        split="train",
        header=_header(),
        recording_excluded=True,
        exclusion_reason="冻结 QC 排除",
    )
    validate_event_table(pd.concat([first, excluded], ignore_index=True))
    partially_excluded = excluded.copy()
    partially_excluded.loc[0, "是否可训练"] = True
    partially_excluded.loc[0, "排除原因"] = ""
    with pytest.raises(ValueError, match="多个文本指纹"):
        validate_event_table(pd.concat([first, partially_excluded], ignore_index=True))


def test_pallier_signal_stage_has_no_model_or_checkpoint_code():
    source = Path(pallier2025.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch.load" not in source
    assert "preload=True" not in source


def test_local_pallier_event_product_and_cli_check(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    event_path = root / "derived/pallier2025/events/events.csv"
    if not event_path.exists():
        pytest.skip("本机尚未物化 Pallier2025 阶段 2A 事件表。")
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    table = pallier2025.load_event_table(event_path)
    assert len(table) == 152560
    assert tuple(table.columns[:14]) == CORE_EVENT_COLUMNS
    assert table["记录编号"].nunique() == 90
    assert table["材料编号"].nunique() == 9
    bad = table[table["记录编号"].eq("pallier2025|sub-09|run-03")]
    assert len(bad) == 1860
    assert not bad["是否可训练"].any()
    event_manifest = build_derived_data.validate_event_product(event_path)
    assert event_manifest["event_count"] == 152560
    assert event_manifest["trainable_event_count"] == 150700
    assert len(event_manifest["audit"]["alignment_audits"]) == 90
    assert event_manifest["audit"]["alignment_summary"] == {
        "exact_recordings": 89,
        "approved_mismatch_recordings": 1,
        "total_substitutions": 0,
        "total_insertions": 2,
        "total_deletions": 2,
    }
    assert event_manifest["run_split"]["assignments"] == {
        "train": ["run-01", "run-02", "run-03", "run-04", "run-05", "run-08", "run-09"],
        "val": ["run-07"],
        "test": ["run-06"],
    }
    result = build_derived_data.check_dataset("pallier2025")
    assert result["status"] == "incomplete"
    assert result["components"]["events"] == "complete"
    assert result["components"]["signals"] in {"missing", "complete"}
    assert result["components"]["text"] == "missing"
    if result["components"]["signals"] == "complete":
        assert result["signal_coverage"] == {
            "expected_recordings": 90,
            "signal_products": 90,
            "missing": 0,
            "extra": 0,
            "channel_count": 306,
            "trainable_windows_in_range": True,
        }
    assert result["test_model_evaluation_performed"] is False
