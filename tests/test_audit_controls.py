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
    "experiments/manifests/conditions.json": "ca14c1fd5276725970e4c1a7042057c7e32ad26b3230e6c3c18af8a918b64c03",
    "experiments/manifests/chineseeeg2/split_manifest.json": "775161256c90cf7cc8eff4e5628f4cf180aa972c0569305994060e6a0609fff7",
    "experiments/manifests/chineseeeg2/vocabulary_N20.json": "ba01a1f7f44dd52193bed5d77cd2d00c32abb75d76770a6b7b7d29691c93cf11",
    "experiments/manifests/chineseeeg2/vocabulary_N50.json": "963f4a63508dd72f5bd38e270a4dc224a87ccedfe28fe48c978a6d48fdfc5850",
    "experiments/manifests/chineseeeg2/vocabulary_N100.json": "0c513b24de8cb9c94cf1d37e3317b91cc66eaa5f3d7b43cf05bd21d74b065ea4",
    "experiments/manifests/chineseeeg2/vocabulary_N150.json": "d1fcc7d21c1ff77afcf812a2d23808265d59abf529fae1d50175a2c0f50dd864",
    "experiments/manifests/chineseeeg2/vocabulary_support.json": "6e237253a753dfc35624aba06020c508a7952d04827e75ac77a9e97104900dff",
    "experiments/manifests/smn4lang/development_manifest_sub01.json": "7649cb3a4cd3415afb739bb09bc615a491767bd0ec09c869f49a1ae01bad497a",
    "experiments/manifests/smn4lang/vocabulary_N20.json": "62992e78c9e8010920dfc51c77b33abd5552dc800f0e1fdcd162f630fb3a1587",
    "experiments/manifests/smn4lang/vocabulary_N50.json": "3e4c6475cb8b0d4865dc693bb72115372417a5d590db48c41b4a96765b750127",
    "experiments/manifests/smn4lang/vocabulary_N100.json": "2416472227ca7ffc3f1aeafeea8e343c2b54a1d30277d3b38df495a45d1fc408",
    "experiments/manifests/smn4lang/vocabulary_N150.json": "1e554cba5d81d7979e338463c91e33bd701535065a549fd783b6f8227733f0f7",
    "experiments/manifests/smn4lang/vocabulary_support.json": "619ce6f72051d7f77ec1699941e715698186294dead2eaf4a163f0405e760a44",
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


def test_existing_control_algorithm_fixture_hashes_do_not_drift():
    assets = _assets()
    expected = {
        "clean_identity_mapping.json": "b66a7225480995c2ccffb20f85f8db62cc31256c9856fcec73a3b090cb2ccd8a",
        "temporal_shift_mapping.json": "2f3dbe5b65c2358c18896cb968e1b1be215ca96c67fbfecb6b5a9a26455b2a2b",
        "donor_swap_seed00.json": "912e0e9c8331abeaf9a6238aee0ef45944160ede449d1cc5cbbc9c1c8d1be9ff",
        "donor_swap_seed19.json": "f7466114012968796e7add0c2755169165566bc7f429bd38dc730f4fd3cf88e3",
        "core_audit_queries.json": "937255ddeb055069c61ea456332727fba12d4dbf9b3958978ddb36000c871077",
    }
    for name, digest in expected.items():
        field = (
            "query_manifest_sha256"
            if name == "core_audit_queries.json"
            else "mapping_sha256"
        )
        assert assets[name][field] == digest


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

    def forward(
        self,
        signals,
        subject_indices=None,
        group_indices=None,
        return_brain_embedding=False,
    ):
        del group_indices
        feature = torch.nn.functional.normalize(
            self.brain_encoder(signals, subject_indices), dim=-1
        )
        if return_brain_embedding:
            return feature, feature
        return feature


class _RecordingContextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.groups = []

    def forward(self, features, group_indices):
        self.inputs.append(features.detach().cpu().clone())
        self.groups.append(group_indices.detach().cpu().clone())
        return features + 1.0


class _FakeContextModel(_FakeWordModel):
    def __init__(self):
        super().__init__()
        self.context_transformer = _RecordingContextTransformer()

    def forward(
        self,
        signals,
        subject_indices=None,
        group_indices=None,
        return_brain_embedding=False,
    ):
        feature = torch.nn.functional.normalize(
            self.brain_encoder(signals, subject_indices), dim=-1
        )
        output = self.context_transformer(feature, group_indices)
        output = torch.nn.functional.normalize(output, dim=-1)
        if return_brain_embedding:
            return output, feature
        return output


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


def test_cached_clean_prediction_matches_direct_forward():
    dataset = _FakeDataset()
    model = _FakeContextModel().eval()
    encoded = evaluate.collect_brain_features(
        model,
        _fake_loader(dataset),
        torch.device("cpu"),
        signal_key="eeg",
        amp=False,
        collect_direct_predictions=True,
    )
    events = controls.prepare_validation_events(dataset.table)
    clean = controls.build_identity_mapping(
        events, dataset="fixture", event_table_sha256="event-table"
    )
    result = evaluate.verify_cached_clean_predictions(
        model,
        encoded,
        clean,
        events["event_id"],
        model_condition="neural_context",
        device=torch.device("cpu"),
        amp=False,
    )
    assert result["allclose"] is True
    assert result["maximum_absolute_difference"] == 0.0


def test_structure_only_zeros_all_context_features_and_preserves_groups():
    model = _FakeContextModel().eval()
    encoded = {
        "brain_features": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "batch_slices": [(0, 3), (3, 6)],
        "sentence_indices": torch.tensor([0, 0, 1, 2, 2, 2]),
    }
    original_groups = encoded["sentence_indices"].clone()
    prediction, audit = evaluate.predict_structure_only_from_cached_features(
        model,
        encoded,
        model_condition="neural_context",
        device=torch.device("cpu"),
        amp=False,
    )
    assert prediction.shape == (6, 4)
    assert all(torch.count_nonzero(value).item() == 0 for value in model.context_transformer.inputs)
    torch.testing.assert_close(
        torch.cat(model.context_transformer.groups), original_groups
    )
    assert audit["all_context_brain_features_zero"] is True
    assert audit["group_length_and_order_unchanged"] is True


def test_word_structure_only_is_explicitly_not_applicable():
    prediction, audit = evaluate.predict_structure_only_from_cached_features(
        _FakeWordModel(),
        {"brain_features": torch.ones(2, 3)},
        model_condition="word",
        device=torch.device("cpu"),
        amp=False,
    )
    assert prediction is None
    assert audit == {
        "status": "not_applicable",
        "reason": "no_context_transformer",
    }


def test_donor_aggregate_uses_all_twenty_seeds():
    per_seed = {}
    for seed in range(20):
        value = seed / 100.0
        per_seed[f"seed-{seed:02d}"] = {
            "N50": {
                "micro_recall_at_1": value,
                "micro_recall_at_10": value + 0.1,
                "macro_recall_at_1": value + 0.2,
                "macro_recall_at_10": value + 0.3,
                "median_rank": 10.0 + seed,
                "mean_reciprocal_rank": value + 0.4,
            }
        }
    aggregate = evaluate.aggregate_donor_results(per_seed)
    assert aggregate["N50"]["retrieval"]["top1"]["count"] == 20
    assert aggregate["N50"]["retrieval"]["top1"]["mean"] == pytest.approx(0.095)


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
