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
    "experiments/manifests/conditions.json": "13a8d0e7b792d85541855afade5b934cb42def2ac8823a9cabe3746c3fab7050",
    "experiments/manifests/chineseeeg2/ovmi_support.json": "bf51055e538c68257d4465cc0787a536cce2b57c8a1df61c36691d77372ff36f",
    "experiments/manifests/chineseeeg2/split_manifest.json": "322b19d16aaa292a806ddcbba67c17d7cc71a075850dc57e4353f541091bcae5",
    "experiments/manifests/chineseeeg2/story_reference.json": "143e26604197a97da67806b72c68b70964afafbd213c7a8a8b3c4b7fc39294ed",
    "experiments/manifests/chineseeeg2/story_reference.provenance.json": "9029e903d1401fc8647a08728929c1f33ab861946a0894b1300e15c842b1bb5d",
    "experiments/manifests/chineseeeg2/vocabulary_N20.json": "bc3d37af058bc8c8e9c30eec287c731e6656f04b8c1a8f57e4aea16e1bf4b307",
    "experiments/manifests/chineseeeg2/vocabulary_N50.json": "ee978eafbc036a15546bda5f6666b0d0d00dac262a96e5f4a3600f0ce8ea5484",
    "experiments/manifests/chineseeeg2/vocabulary_N100.json": "0114918d7c70ced46bb30307f8f0bba0673b6a353100ac229c1f144ba413a273",
    "experiments/manifests/chineseeeg2/vocabulary_N150.json": "fdad18e16d35556e488197d9c2327121782c4a87e99e6d7cdf52d8fadeacdaa1",
    "experiments/manifests/smn4lang/development_manifest_sub01.json": "4389c31e288419f452ddab0774f7cee993210e83e663348c68c4946022a69603",
    "experiments/manifests/smn4lang/ovmi_support.json": "9ccf79a6ddb15bde4f79d88884eda5e7c70ec09faa7b05f2d071caaa4d50a697",
    "experiments/manifests/smn4lang/story_reference.json": "c733d0bb02566ebb858e21aa5cbda1885378bf7a408a99489c34bb64e5c0d5b2",
    "experiments/manifests/smn4lang/story_reference.provenance.json": "7f2bc8fc2104e738852413681e247bcafc992fff6975ac2a2606c534b0d97b15",
    "experiments/manifests/smn4lang/vocabulary_N20.json": "a19712efdd65333433346c3780718c81a3aebb3409786cbb08460f00dadf63e0",
    "experiments/manifests/smn4lang/vocabulary_N50.json": "f1103e2487fdccc0088dcfb8830905339cf10f0d2a55ded123cea36465c9efcf",
    "experiments/manifests/smn4lang/vocabulary_N100.json": "83d905894249f01c6852ee0014d3f9bdcae98575121dc2c796b9b32261e42cfa",
    "experiments/manifests/smn4lang/vocabulary_N150.json": "ccdb955567097f024a2d0f0b4b89945dc2035feebaecd40575d19e6f4646f363",
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
                "ovmi_story": {"available": True, "score_bits": value + 0.5},
            }
        }
    aggregate = evaluate.aggregate_donor_results(per_seed)
    assert aggregate["N50"]["retrieval"]["top1"]["count"] == 20
    assert aggregate["N50"]["retrieval"]["top1"]["mean"] == pytest.approx(0.095)
    assert aggregate["N50"]["ovmi_story"]["score_bits"]["count"] == 20


def test_ovmi_language_is_supplied_by_dataset_adapter(monkeypatch):
    observed = {}

    def fake_ovmi(true_words, predicted_words, vocabulary, config):
        del true_words, predicted_words, vocabulary
        observed["language"] = config["language"]
        return {"available": True, "score_bits": 0.0}

    monkeypatch.setattr(evaluate, "full_ovmi_metrics", fake_ovmi)
    encoded = {
        "event_ids": ["a", "b"],
        "target_embeddings": torch.eye(2),
        "words": ["a", "b"],
    }
    result = evaluate.evaluate_control(
        torch.eye(2),
        encoded,
        ["a", "b"],
        {2: {"vocabulary": ["a", "b"], "manifest_sha256": "sha"}},
        None,
        {"a": 1, "b": 1},
        dataset_name="pallier2025",
        language="fr",
    )
    assert observed["language"] == "fr"
    assert result["N2"]["ovmi_story"]["available"] is True


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
