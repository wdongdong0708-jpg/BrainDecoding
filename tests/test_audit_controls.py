"""validation-only 神经证据替换映射与统一评价测试。"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from braindecoding.audit import controls
from braindecoding.audit import evaluate
from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL_FILE_SHA256 = {
    "experiments/manifests/conditions.json": "a8b1b71402756ee51f9883d5f0d6fbd8bbdd0cf818d11ba189923d433d18212d",
    "experiments/manifests/chineseeeg2/ovmi_support.json": "d6c93ef38445f3b36f248bb36652150cf9fd5fe9c3387f542dbe78b8933a7349",
    "experiments/manifests/chineseeeg2/split_manifest.json": "202cc6a5b5d1fdc36437917dea5f8cfca7db9e10877e0ae7ce888c2e4b422bc9",
    "experiments/manifests/chineseeeg2/story_reference.json": "143e26604197a97da67806b72c68b70964afafbd213c7a8a8b3c4b7fc39294ed",
    "experiments/manifests/chineseeeg2/story_reference.provenance.json": "ae8c19434af0d2326d96fb3130d9500c8be9836b3c6a28e17b919ad8b65dbb9a",
    "experiments/manifests/chineseeeg2/vocabulary_N20.json": "c2dc3c08cc15c198f0deb8f09955ec3262498f390bf6be1742784e4071e006ce",
    "experiments/manifests/chineseeeg2/vocabulary_N50.json": "278a75892d68242dd0bb71ef46f40ad8633b8222a185f80c19bdb68dfb3d53fb",
    "experiments/manifests/chineseeeg2/vocabulary_N100.json": "cfc8bda263eaa56c67a9422d416ff9e79716d3ff05fa53b01f227e085262910a",
    "experiments/manifests/chineseeeg2/vocabulary_N150.json": "15476e49a92698c3a585f40a034e51f4115eae8d0cda057fcce5cddd56d630ad",
    "experiments/manifests/smn4lang/development_manifest_sub01.json": "79255bb8d573c0790ff02f5320c7c9c168cd68dfc289e7e9304f76341d7b49ac",
    "experiments/manifests/smn4lang/ovmi_support.json": "3f12f4e8030ba4669103d10a6599b4ebdbb749699ee596d040ab331f18edb62c",
    "experiments/manifests/smn4lang/story_reference.json": "c733d0bb02566ebb858e21aa5cbda1885378bf7a408a99489c34bb64e5c0d5b2",
    "experiments/manifests/smn4lang/story_reference.provenance.json": "164c48316c8270f2b7e2ca9ce408509b5ad9d095e7e78127d1ae48ca954d360d",
    "experiments/manifests/smn4lang/vocabulary_N20.json": "c233e23dc08d81c55113afc8d9cda5e3ebb1a2b6e39d245810e2cd0c3fc531e7",
    "experiments/manifests/smn4lang/vocabulary_N50.json": "eae4434bbbee659ae226ada1721eddf5f0013c5bda20b58287ba8b2f6690cb72",
    "experiments/manifests/smn4lang/vocabulary_N100.json": "e8583306dfe93e089c25766fbb71a8f654de6af4ddef162854e7156b039c4090",
    "experiments/manifests/smn4lang/vocabulary_N150.json": "950908b6b1c2c93f10a4aa7f2b68b9219d0df84fd802ce59baa07fcea7fe926d",
}


def _events(words=("甲", "乙", "丙", "丁", "戊", "己")):
    size = len(words)
    return pd.DataFrame(
        {
            "event_id": [f"event-{index}" for index in range(size)],
            "subject_id": ["sub-01"] * size,
            "recording_id": ["recording-01"] * size,
            "normalized_word": list(words),
            "sentence_uid": ["context-01"] * size,
            "split": ["val"] * size,
            "is_trainable": [True] * size,
            "window_complete": [True] * size,
            "window_start_seconds": np.arange(size, dtype=float) * 2.0,
            "window_stop_seconds": np.arange(size, dtype=float) * 2.0 + 1.0,
        }
    )


def _assets(words=("甲", "乙", "丙", "丁", "戊", "己"), status="validation_only"):
    return controls.build_control_assets(
        _events(words),
        dataset="fixture",
        event_table_sha256="event-table",
        status=status,
    )


def test_clean_mapping_is_identity():
    mapping = _assets()["clean_identity_mapping.json"]
    assert all(
        row["target_event_id"] == row["source_event_id"]
        for row in mapping["mappings"]
    )


def test_temporal_mapping_is_deterministic_and_uses_no_word_labels():
    first = _assets()["temporal_shift_mapping.json"]
    second = _assets(("词一", "词二", "词三", "词四", "词五", "词六"))[
        "temporal_shift_mapping.json"
    ]
    assert first == second
    assert first["algorithm"]["uses_word_labels"] is False


def test_temporal_mapping_stays_within_subject_recording_and_does_not_overlap():
    mapping = _assets()["temporal_shift_mapping.json"]
    assert mapping["eligible_count"] == 6
    for row in mapping["mappings"]:
        assert row["source_event_id"] != row["target_event_id"]
        assert row["subject"] == "sub-01"
        assert row["recording"] == "recording-01"
        assert (
            row["source_window_stop"] <= row["target_window_start"]
            or row["target_window_stop"] <= row["source_window_start"]
        )


def test_temporal_mapping_hash_and_serialization_are_stable():
    first = _assets()["temporal_shift_mapping.json"]
    second = _assets()["temporal_shift_mapping.json"]
    assert first["mapping_sha256"] == second["mapping_sha256"]
    assert controls.json_bytes(first) == controls.json_bytes(second)
    controls.validate_payload_sha256(first, "mapping_sha256")


def test_donor_mapping_obeys_subject_recording_word_and_window_contract():
    mapping = _assets()["donor_swap_seed00.json"]
    for row in mapping["mappings"]:
        assert row["subject"] == "sub-01"
        assert row["recording"] == "recording-01"
        assert row["source_event_id"] != row["target_event_id"]
        assert row["source_standard_word"] != row["target_standard_word"]
        assert row["source_window_stop"] <= row["target_window_start"] or (
            row["target_window_stop"] <= row["source_window_start"]
        )
        assert (row["source_window_stop"] - row["source_window_start"]) == pytest.approx(
            row["target_window_stop"] - row["target_window_start"]
        )


def test_all_twenty_donor_seeds_are_reproducible():
    first = _assets()
    second = _assets()
    hashes = []
    for seed in range(20):
        name = f"donor_swap_seed{seed:02d}.json"
        assert first[name] == second[name]
        assert first[name]["seed"] == seed
        hashes.append(first[name]["mapping_sha256"])
    assert len(hashes) == 20


def test_common_support_is_identical_for_all_three_conditions():
    assets = _assets()
    core = assets["core_audit_queries.json"]
    query_ids = set(core["event_ids"])
    assert query_ids == {
        row["target_event_id"]
        for row in assets["clean_identity_mapping.json"]["mappings"]
    }
    assert query_ids == {
        row["target_event_id"]
        for row in assets["temporal_shift_mapping.json"]["mappings"]
    }
    for seed in range(20):
        assert query_ids == {
            row["target_event_id"]
            for row in assets[f"donor_swap_seed{seed:02d}.json"]["mappings"]
        }


def test_word_and_context_consume_the_same_mapping_without_moving_positions():
    assets = _assets()
    mapping = assets["donor_swap_seed00.json"]
    event_ids = _events()["event_id"].tolist()
    encoded = {
        "brain_features": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "event_ids": event_ids,
        "sentence_indices": torch.tensor([0, 0, 0, 1, 1, 1]),
    }
    original_groups = encoded["sentence_indices"].clone()
    word_features, word_audit = evaluate.apply_feature_mapping(
        encoded, mapping, event_ids
    )
    context_features, context_audit = evaluate.apply_feature_mapping(
        encoded, mapping, event_ids
    )
    torch.testing.assert_close(word_features, context_features)
    torch.testing.assert_close(encoded["sentence_indices"], original_groups)
    assert word_audit == context_audit
    assert word_audit["context_group_indices_unchanged"] is True
    assert word_audit["target_positions_unchanged"] is True


class _FakeBrainEncoder(nn.Module):
    def forward(self, signals, subject_indices):
        flattened = signals.flatten(start_dim=1)
        return flattened + subject_indices[:, None].float() * 0.01


class _FakeWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.brain_encoder = _FakeBrainEncoder()
        self.context_transformer = None
        self.context_mode = "grouped"


class _FakeDataset:
    def __init__(self):
        self.table = _events().iloc[:4].reset_index(drop=True)
        self.signals = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, -1.0]],
            ]
        )

    def __getitem__(self, index):
        row = self.table.iloc[index]
        return {
            "eeg": self.signals[index],
            "subject_index": torch.tensor(0),
            "event_id": row["event_id"],
        }


def _fake_loader(dataset):
    return [
        {
            "eeg": dataset.signals,
            "text_embedding": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
            ),
            "subject_index": torch.zeros(4, dtype=torch.long),
            "sentence_index": torch.tensor([0, 0, 1, 1]),
            "word": ["甲", "乙", "丙", "丁"],
            "event_id": dataset.table["event_id"].tolist(),
            "recording_id": dataset.table["recording_id"].tolist(),
        }
    ]


def test_cached_donor_feature_matches_direct_window_forward():
    dataset = _FakeDataset()
    model = _FakeWordModel().eval()
    encoded = evaluate.collect_brain_features(
        model, _fake_loader(dataset), torch.device("cpu"), signal_key="eeg", amp=False
    )
    events = controls.prepare_validation_events(dataset.table)
    candidates = controls.donor_candidates(events)
    mapping = controls.build_donor_swap_mapping(
        events,
        candidates,
        dataset="fixture",
        event_table_sha256="event-table",
        seed=0,
    )
    result = evaluate.verify_cached_donor_features(
        model,
        dataset,
        encoded,
        mapping,
        torch.device("cpu"),
        signal_key="eeg",
        amp=False,
    )
    assert result["equivalent"] is True
    assert result["maximum_absolute_difference"] == 0.0


def test_clean_cached_runner_matches_existing_retrieval_exactly():
    dataset = _FakeDataset()
    model = _FakeWordModel().eval()
    encoded = evaluate.collect_brain_features(
        model, _fake_loader(dataset), torch.device("cpu"), signal_key="eeg", amp=False
    )
    events = controls.prepare_validation_events(dataset.table)
    clean = controls.build_identity_mapping(
        events, dataset="fixture", event_table_sha256="event-table"
    )
    predictions, _ = evaluate.predict_from_cached_features(
        model,
        encoded,
        clean,
        events["event_id"],
        model_condition="word",
        device=torch.device("cpu"),
        amp=False,
    )
    vocabulary = ["甲", "乙", "丙", "丁"]
    actual, _ = evaluate.fixed_vocabulary_metrics_for_queries(
        predictions,
        encoded,
        events["event_id"],
        vocabulary,
        vocabulary_name="fixture",
    )
    expected = fixed_vocabulary_retrieval_metrics(
        predictions,
        encoded["target_embeddings"],
        encoded["words"],
        vocabulary,
        vocabulary_name="fixture",
    )
    assert actual == expected


def test_runner_rejects_test_before_any_neural_access():
    with pytest.raises(ValueError, match="禁止访问 test"):
        controls.prepare_validation_events(_events(), split="test")


def test_smn_mapping_assets_are_explicitly_development_only():
    assets = _assets(status="development_only")
    assert all(payload["status"] == "development_only" for payload in assets.values())


def test_frozen_protocol_manifest_files_are_unchanged():
    for relative_path, expected in FROZEN_PROTOCOL_FILE_SHA256.items():
        digest = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == expected
