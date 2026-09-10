import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import savemat

from datasets import SMN4Lang as dataset_module
from datasets.SMN4Lang import (
    _embedding_signature,
    ensure_repository_gpt2_word_prototypes,
    meg_word_times,
    processed_recording_path,
    repository_gpt2_embedding_signature,
    script_sentence_indices,
    split_for_run,
    training_event_mask,
)
from braindecoding.training.word import SentenceBatchSampler
from tasks.word_decoding.SMN4Lang.evaluate import evaluate_checkpoint
from tasks.word_decoding.SMN4Lang.train import load_config, run_training
from models import build_brain_embedding_model
from tasks.word_decoding.SMN4Lang.train import (
    load_pretrained_brain_encoder,
    set_brain_encoder_trainable,
)


CHANNEL_NAMES = [
    "MEG0111",
    "MEG0112",
    "MEG0113",
    "MEG0121",
    "MEG0122",
    "MEG0123",
]


def _make_recording_files(root):
    meg_dir = root / "derivatives" / "preprocessed_data" / "sub-01" / "MEG"
    meg_dir.mkdir(parents=True)
    stem = "sub-01_task-RDR_run-1"
    fif_path = meg_dir / f"{stem}_meg.fif"
    fif_path.write_bytes(b"synthetic-fif-placeholder")
    (meg_dir / f"{stem}_meg.json").write_text(
        json.dumps({"SamplingFrequency": 100.0, "RecordingDuration": 20.0}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "name": CHANNEL_NAMES,
            "type": ["MEGMAG", "MEGGRADPLANAR", "MEGGRADPLANAR"] * 2,
            "status": ["good"] * len(CHANNEL_NAMES),
        }
    ).to_csv(meg_dir / f"{stem}_channels.tsv", sep="\t", index=False)
    return fif_path


def _dataset_config(root):
    return {
        "name": "SMN4Lang",
        "root": str(root),
        "subjects": ["sub-01"],
        "split": {
            "train_runs": [1],
            "val_runs": [2],
            "test_runs": [3],
        },
        "source_sampling_rate_hz": 100,
        "target_sampling_rate_hz": 50,
        "window_seconds": 1,
        "baseline_seconds": 0,
        "filter_low_hz": None,
        "filter_high_hz": None,
        "filter_order": 2,
        "scaler": "RobustScaler",
        "clamp": 5,
        "materialization_channel_chunk": 2,
    }


def test_frozen_split_and_meg_clock_transform():
    assert split_for_run(1) == "train"
    assert split_for_run(51) == "val"
    assert split_for_run(60) == "test"
    with pytest.raises(ValueError):
        split_for_run(61)
    starts, stops = meg_word_times([11.28], [11.45], 13.726)
    assert starts[0] == pytest.approx(14.3955)
    assert stops[0] - starts[0] == pytest.approx(0.17)


def test_training_uses_all_complete_words_not_only_top50():
    frame = pd.DataFrame(
        {
            "normalized_word": ["的", "稀有词", "", "窗口不完整"],
            "in_smn4lang50": [True, False, False, False],
            "window_complete": [True, True, True, False],
        }
    )
    assert training_event_mask(frame).tolist() == [True, True, False, False]


def test_conv_only_config_changes_only_transformer_switch_and_output_dir():
    project_root = Path(__file__).resolve().parents[1]
    transformer_config = load_config(project_root / "configs" / "SMN4Lang.yaml")
    conv_only_config = load_config(
        project_root / "configs" / "SMN4Lang_conv_only.yaml"
    )

    assert transformer_config["model"]["use_transformer"] is True
    assert conv_only_config["model"]["use_transformer"] is False
    assert (
        transformer_config["training"]["output_dir"]
        != conv_only_config["training"]["output_dir"]
    )
    transformer_config["model"]["use_transformer"] = False
    transformer_config["training"]["output_dir"] = conv_only_config["training"][
        "output_dir"
    ]
    for section in (
        "dataset",
        "cache",
        "text_embedding",
        "model",
        "loss",
        "training",
        "evaluation",
    ):
        assert transformer_config[section] == conv_only_config[section]


def test_one_second_cnn_warm_start_changes_only_training_stages_and_output():
    project_root = Path(__file__).resolve().parents[1]
    baseline = load_config(project_root / "configs" / "SMN4Lang_1s.yaml")
    staged = load_config(
        project_root / "configs" / "SMN4Lang_1s_cnn_warm_start.yaml"
    )

    assert staged["training"]["freeze_brain_encoder_updates"] == 960
    assert staged["training"]["max_updates"] == 6400
    assert staged["model"] == baseline["model"]
    for section in ("dataset", "cache", "text_embedding", "loss", "evaluation"):
        assert staged[section] == baseline[section]


def test_pretrained_checkpoint_loads_only_brain_encoder(tmp_path):
    positions = np.asarray(
        [[index, index % 2, 1.0] for index in range(6)], dtype=np.float32
    )
    conv = {
        "merger_channels": 4,
        "merger_position_dimension": 8,
        "merger_dropout": 0,
        "initial_linear": 6,
        "hidden": 4,
        "depth": 2,
        "kernel_size": 3,
        "dilation_growth": 2,
        "dilation_period": 2,
        "dropout_input": 0,
        "batch_norm": False,
        "gelu": True,
        "skip": True,
        "glu_every": 0,
        "temporal_attention_hidden": 4,
    }
    source_model_config = {
        "embedding_dimension": 8,
        "use_transformer": False,
        "conv": conv,
    }
    target_model_config = {
        **source_model_config,
        "use_transformer": True,
        "transformer": {"depth": 1, "heads": 2},
    }
    source = build_brain_embedding_model(6, positions, 1, source_model_config)
    target = build_brain_embedding_model(6, positions, 1, target_model_config)
    with torch.no_grad():
        for parameter in source.brain_encoder.parameters():
            parameter.fill_(0.125)
    transformer_before = {
        key: value.detach().clone()
        for key, value in target.context_transformer.state_dict().items()
    }
    dataset_contract = {
        "window_seconds": 1.0,
        "target_sampling_rate_hz": 50,
        "split": {"train_runs": [1], "val_runs": [2], "test_runs": [3]},
        "training_vocabulary_policy": "all_nonempty_complete_window_words",
        "evaluation_vocabulary_policy": "frozen_train_only_top50",
    }
    text_config = {"model_name": "synthetic", "embedding_dimension": 8}
    checkpoint_path = tmp_path / "cnn.pt"
    torch.save(
        {
            "task": "word_decoding/SMN4Lang",
            "epoch": 3,
            "model_state": source.state_dict(),
            "model_config": source_model_config,
            "text_embedding_config": text_config,
            "dataset_contract": dataset_contract,
            "channel_names": CHANNEL_NAMES,
            "channel_positions": positions.tolist(),
            "metrics": {"retrieval_acc10_vocab=smn4lang50_macro": 0.3},
        },
        checkpoint_path,
    )
    config = {
        "dataset": dataset_contract,
        "text_embedding": text_config,
        "model": target_model_config,
    }

    audit = load_pretrained_brain_encoder(
        target, checkpoint_path, config, CHANNEL_NAMES, positions
    )

    for key, value in source.brain_encoder.state_dict().items():
        torch.testing.assert_close(target.brain_encoder.state_dict()[key], value)
    for key, value in transformer_before.items():
        torch.testing.assert_close(target.context_transformer.state_dict()[key], value)
    assert audit["loaded_loss_state"] is False
    assert audit["loaded_transformer_state"] is False
    set_brain_encoder_trainable(target, False)
    assert not any(parameter.requires_grad for parameter in target.brain_encoder.parameters())
    assert target.brain_encoder.training is False
    set_brain_encoder_trainable(target, True)
    assert all(parameter.requires_grad for parameter in target.brain_encoder.parameters())


def test_repository_gpt2_builds_train_only_word_prototypes(tmp_path):
    embedding_dir = tmp_path / "embeddings"
    embedding_dir.mkdir()
    data = np.zeros((25, 4, 4), dtype=np.float32)
    data[24] = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [3.0, 4.0, 5.0, 6.0],
            [9.0, 9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    savemat(embedding_dir / "story_1.mat", {"data": data})
    event_table = pd.DataFrame(
        {
            "split": ["train", "train", "train", "val"],
            "is_trainable": [True, True, True, True],
            "run": [1, 1, 1, 1],
            "word_index": [0, 1, 2, 3],
            "normalized_word": ["甲", "乙", "甲", "验证词"],
        }
    )
    dataset_config = {
        "root": str(tmp_path),
        "snapshot": "synthetic",
        "split": {"train_runs": [1]},
    }
    text_config = {
        "source": "smn4lang_repository_gpt2",
        "relative_dir": "embeddings",
        "filename_pattern": "story_{run}.mat",
        "layer_index": 24,
        "layer_count": 25,
        "embedding_dimension": 4,
        "aggregation": "train_mean_by_normalized_word",
    }
    cache_path = tmp_path / "prototypes.npz"
    ensure_repository_gpt2_word_prototypes(
        event_table, dataset_config, text_config, cache_path
    )
    with np.load(cache_path, allow_pickle=False) as payload:
        words = payload["words"].astype(str).tolist()
        embeddings = dict(zip(words, payload["embeddings"]))
        counts = dict(zip(words, payload["counts"].tolist()))
    assert set(words) == {"甲", "乙"}
    np.testing.assert_allclose(embeddings["甲"], [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(embeddings["乙"], [4.0, 3.0, 2.0, 1.0])
    assert counts == {"乙": 1, "甲": 2}
    metadata = json.loads(
        cache_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    assert metadata["candidate_provenance"] == "train_runs_only"
    assert metadata["signature"] == repository_gpt2_embedding_signature(
        dataset_config, text_config
    )

    full_cache_bytes = cache_path.read_bytes()
    ensure_repository_gpt2_word_prototypes(
        event_table.iloc[:1], dataset_config, text_config, cache_path
    )
    assert cache_path.read_bytes() == full_cache_bytes


def test_gpt2_default_config_uses_1024_dimensions_and_compatible_heads():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "SMN4Lang_gpt2.yaml")
    assert config["text_embedding"]["source"] == "smn4lang_repository_gpt2"
    assert config["text_embedding"]["layer_index"] == 24
    assert config["model"]["embedding_dimension"] == 1024
    assert 1024 % config["model"]["transformer"]["heads"] == 0


def test_script_sentences_and_sampler_keep_batches_bounded(tmp_path):
    script_path = tmp_path / "story.txt"
    script_path.write_text("我们会说。\n教育很好。\n", encoding="utf-8")
    indices, audit = script_sentence_indices(
        ["我们", "会", "说", "教育", "很好"],
        script_path,
        max_character_mismatches=0,
    )
    assert indices.tolist() == [0, 0, 0, 1, 1]
    assert audit["sentence_count"] == 2
    assert audit["character_mismatch_count"] == 0

    sampler = SentenceBatchSampler(
        ["one-long-story"] * 277,
        batch_size=128,
        shuffle=False,
        seed=0,
    )
    batches = list(sampler)
    assert [len(batch) for batch in batches] == [128, 128, 21]
    assert max(map(len, batches)) <= 128
    assert [index for batch in batches for index in batch] == list(range(277))
    assert len(sampler) == 3


def test_recording_materialization_uses_selected_meg_channels(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    fif_path = _make_recording_files(root)
    random = np.random.default_rng(0)
    signal = random.normal(size=(len(CHANNEL_NAMES), 2000)).astype("float32")

    class FakeRaw:
        info = {"sfreq": 100.0}
        ch_names = CHANNEL_NAMES

        def get_data(self, picks):
            indices = [self.ch_names.index(name) for name in picks]
            return signal[indices]

        def close(self):
            return None

    monkeypatch.setattr(dataset_module, "_read_raw_fif", lambda path: FakeRaw())
    config = _dataset_config(root)
    output = dataset_module.materialize_recording_cache(
        fif_path, config, tmp_path / "cache"
    )
    data = np.load(output)
    assert data.shape == (6, 1000)
    assert data.dtype == np.float32
    assert np.isfinite(data).all()


def test_one_epoch_training_pipeline_on_synthetic_cache(tmp_path):
    root = tmp_path / "dataset"
    fif_path = _make_recording_files(root)
    dataset_config = _dataset_config(root)
    random = np.random.default_rng(1)
    cache_dir = tmp_path / "meg_cache"
    cache_dir.mkdir()
    cache_path = processed_recording_path(fif_path, dataset_config, cache_dir)
    recording = random.normal(size=(6, 1000)).astype("float32")
    np.save(cache_path, recording)
    signature = dataset_module._preprocessing_signature(dataset_config, fif_path)
    cache_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": "materialized",
                "signature": signature,
                "shape": list(recording.shape),
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )

    rows = []
    for index, (split, word, start) in enumerate(
        [
            ("train", "的", 5),
            ("train", "是", 10),
            ("train", "的", 15),
            ("train", "是", 20),
            ("val", "的", 25),
            ("val", "是", 30),
        ]
    ):
        rows.append(
            {
                "subject_id": "sub-01",
                "split": split,
                "normalized_word": word,
                "sentence_uid": f"{split}-story",
                "source_sentence_uid": f"{split}-story",
                "context_chunk_index": 0,
                "context_grouping": "script_sentence_then_contiguous_chunks",
                "fif_relpath": fif_path.relative_to(root).as_posix(),
                "window_start_target_sample": start,
                "is_trainable": True,
                "run": 1 if split == "train" else 2,
                "onset_seconds": start / 50,
                "word_index": index,
                "target_sample_count": 50,
                "event_id": f"event-{index}",
                "recording_id": "synthetic-recording",
            }
        )
    event_path = tmp_path / "events.csv"
    pd.DataFrame(rows).to_csv(event_path, index=False)

    embedding_path = tmp_path / "embeddings.npz"
    np.savez(
        embedding_path,
        words=np.asarray(["的", "是"]),
        embeddings=random.normal(size=(2, 8)).astype("float32"),
    )
    text_config = {
        "model_name": "synthetic-mengzi",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "embedding_dimension": 8,
    }
    embedding_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": "materialized",
                "signature": _embedding_signature(text_config),
                "word_count": 2,
                "embedding_dimension": 8,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "dataset": dataset_config,
        "cache": {
            "event_table": str(event_path),
            "meg_dir": str(cache_dir),
            "text_embeddings": str(embedding_path),
        },
        "text_embedding": text_config,
        "model": {
            "embedding_dimension": 8,
            "use_transformer": False,
            "conv": {
                "merger_channels": 4,
                "merger_position_dimension": 8,
                "merger_dropout": 0,
                "initial_linear": 6,
                "hidden": 4,
                "depth": 2,
                "kernel_size": 3,
                "dilation_growth": 2,
                "dilation_period": 2,
                "dropout_input": 0,
                "batch_norm": False,
                "gelu": True,
                "skip": True,
                "glu_every": 0,
                "temporal_attention_hidden": 4,
            },
        },
        "loss": {
            "initial_temperature": 10,
            "initial_bias": -10,
            "identical_candidates_threshold": 0.999,
        },
        "training": {
            "output_dir": str(tmp_path / "outputs"),
            "seed": 0,
            "device": "cpu",
            "batch_size": 4,
            "num_workers": 0,
            "epochs": 1,
            "patience": 1,
            "max_updates": 3,
            "minimum_updates_before_early_stopping": 3,
            "scheduler_total_updates": 6,
            "learning_rate": 1e-4,
            "weight_decay": 0,
            "minimum_learning_rate": 0,
            "max_grad_norm": 0,
            "amp": False,
            "top_ks": [1, 10],
            "smoke_rows": 8,
        },
        "provenance": {},
    }
    summary = run_training(config, smoke=True, save=True)
    assert summary["status"] == "smoke_completed"
    assert summary["train_rows"] == 4
    assert summary["validation_rows"] == 2
    assert summary["test_meg_opened"] is False
    assert summary["optimizer_updates"] == 1
    assert summary["target_updates"] == 1
    assert summary["batch_size_limit"] == 4
    assert summary["stop_reason"] == "update_budget_reached"
    evaluation = evaluate_checkpoint(
        config,
        tmp_path / "outputs" / "best.pt",
        split="val",
        smoke=True,
        save=False,
    )
    assert evaluation["status"] == "smoke_evaluation_completed"
    assert evaluation["test_meg_opened"] is False
