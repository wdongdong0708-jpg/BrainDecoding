import json

import h5py
import numpy as np
import pandas as pd
import torch

from braindecoding.data.text import normalize_word, text_embedding_signature
from braindecoding.data.libribrain import (
    materialize_recording_cache,
    processed_recording_path,
)
from losses import build_siglip_loss
from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval_metrics
from models import build_brain_embedding_model
from tasks.word_decoding.LibriBrain100.evaluate import evaluate_checkpoint
from tasks.word_decoding.LibriBrain100.train import load_config, run_training


def test_word_cleanup_matches_source_contract():
    assert normalize_word(" We're! ") == "we're"
    assert normalize_word("co-operate") == "co-operate"


def test_active_word_and_context_configs_share_scientific_data_contract(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    word = load_config(
        "configs/word_decoding/libribrain100/sub0/main_word.yaml"
    )
    context = load_config(
        "configs/word_decoding/libribrain100/sub0/main_context.yaml"
    )
    assert context["dataset"] == word["dataset"]
    assert context["cache"] == word["cache"]
    assert context["text_embedding"] == word["text_embedding"]
    assert context["loss"] == word["loss"]
    assert context["evaluation"] == word["evaluation"]
    assert word["model"]["use_transformer"] is False
    assert context["model"]["use_transformer"] is True
    assert "pretrained_brain_encoder_checkpoint" not in context["training"]


def test_recording_materialization(tmp_path):
    source = tmp_path / "recording.h5"
    random = np.random.default_rng(0)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("data", data=random.normal(size=(4, 1000)).astype("float32"))
        handle.attrs["sample_frequency"] = 100.0
    config = {
        "source_sampling_rate_hz": 100,
        "target_sampling_rate_hz": 50,
        "filter_low_hz": 1.0,
        "filter_high_hz": 20.0,
        "filter_order": 2,
        "scaler": "RobustScaler",
    }
    output = materialize_recording_cache(source, config, tmp_path / "cache")
    assert output == processed_recording_path(source, config, tmp_path / "cache")
    data = np.load(output)
    assert data.shape == (4, 500)
    assert data.dtype == np.float32
    assert np.isfinite(data).all()


def test_small_libribrain_model_loss_and_metric():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = {
        "embedding_dimension": 8,
        "use_transformer": True,
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
        "transformer": {
            "depth": 1,
            "heads": 2,
            "attention_dropout": 0,
            "feedforward_dropout": 0,
        },
    }
    model = build_brain_embedding_model(6, positions, 1, config)
    estimates = model(
        torch.randn(3, 6, 20),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([0, 0, 1]),
    )
    assert estimates.shape == (3, 8)
    assert torch.allclose(
        torch.linalg.vector_norm(estimates, dim=1), torch.ones(3), atol=1e-5
    )

    targets = torch.randn(3, 8)
    targets[2] = targets[0]
    loss = build_siglip_loss()(estimates, targets)
    loss.backward()
    assert torch.isfinite(loss)

    metrics = fixed_vocabulary_retrieval_metrics(
        targets.detach().numpy(),
        targets.detach().numpy(),
        ["is", "the", "is"],
        ["is", "the"],
        top_ks=(1,),
    )
    assert metrics["macro_recall_at_1"] == 1.0
    assert metrics["observed_vocabulary_size"] == 2


def test_one_epoch_training_pipeline_on_synthetic_cache(tmp_path):
    root = tmp_path / "dataset"
    h5_path = root / "Sherlock1" / "derivatives" / "serialised" / "recording.h5"
    h5_path.parent.mkdir(parents=True)
    channel_names = [
        "MEG0111",
        "MEG0112",
        "MEG0113",
        "MEG0121",
        "MEG0122",
        "MEG0123",
    ]
    random = np.random.default_rng(1)
    with h5py.File(h5_path, "w") as handle:
        handle.create_dataset(
            "data", data=random.normal(size=(6, 1000)).astype("float32")
        )
        handle.attrs["sample_frequency"] = 100.0
        handle.attrs["channel_names"] = ", ".join(channel_names)
        handle.attrs["channel_types"] = ", ".join(["mag"] * len(channel_names))

    rows = []
    for index, (split, word, start) in enumerate(
        [
            ("train", "is", 5),
            ("train", "the", 10),
            ("train", "is", 15),
            ("train", "the", 20),
            ("val", "is", 25),
            ("val", "the", 30),
        ]
    ):
        rows.append(
            {
                "split": split,
                "normalized_word": word,
                "sentence_uid": f"{split}-sentence",
                "h5_relpath": h5_path.relative_to(root).as_posix(),
                "window_start_target_sample": start,
                "is_trainable": True,
                "session": 1 if split == "train" else 2,
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
    embeddings = random.normal(size=(2, 8)).astype("float32")
    np.savez(
        embedding_path,
        words=np.asarray(["is", "the"]),
        embeddings=embeddings,
    )
    text_config = {
        "model_name": "t5-large",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "embedding_dimension": 8,
    }
    embedding_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": "materialized",
                "signature": text_embedding_signature(text_config),
                "word_count": 2,
                "embedding_dimension": 8,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "name": "LibriBrain100",
            "root": str(root),
            "task": "Sherlock1",
            "split": {
                "train_sessions": [1],
                "val_sessions": [2],
                "test_sessions": [3],
            },
            "source_sampling_rate_hz": 100,
            "target_sampling_rate_hz": 50,
            "window_seconds": 1,
            "baseline_seconds": 0.1,
            "filter_low_hz": None,
            "filter_high_hz": None,
            "filter_order": 2,
            "scaler": "RobustScaler",
            "clamp": 5,
        },
        "cache": {
            "event_table": str(event_path),
            "meg_dir": str(tmp_path / "meg_cache"),
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
    assert summary["test_eeg_opened"] is False
    evaluation = evaluate_checkpoint(
        config,
        tmp_path / "outputs" / "best.pt",
        split="val",
        smoke=True,
        save=False,
    )
    assert evaluation["status"] == "smoke_evaluation_completed"
    assert evaluation["split"] == "val"
    assert evaluation["test_eeg_opened"] is False
