import math

import numpy as np
import torch

from braindecoding.models import (
    BrainEmbeddingModel,
    SpatialChannelMerger,
    SubjectLinear,
    build_strided_conv_encoder,
)


def test_singleton_context_ignores_external_group_assignments():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = {
        "embedding_dimension": 8,
        "use_transformer": True,
        "context_mode": "singleton",
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
    model = BrainEmbeddingModel(positions, subject_count=1, config=config).eval()
    signals = torch.randn(3, 6, 20)
    subjects = torch.zeros(3, dtype=torch.long)
    grouped = model(signals, subjects, torch.zeros(3, dtype=torch.long))
    separated = model(signals, subjects, torch.arange(3))
    assert torch.allclose(grouped, separated, atol=1e-6)


def test_brain_embedding_model_migrates_legacy_subject_projection_orientation():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    config = {
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
    }
    source = BrainEmbeddingModel(positions, subject_count=1, config=config)
    current_state = source.state_dict()
    legacy_state = {key: value.detach().clone() for key, value in current_state.items()}
    coefficients = legacy_state.pop("brain_encoder.channel_merger.coefficients")
    legacy_state.pop("brain_encoder.channel_merger.basis")
    legacy_state["brain_encoder.channel_merger.heads"] = coefficients
    legacy_state["brain_encoder.channel_merger.channel_positions"] = (
        source.brain_encoder.channel_merger.channel_positions.detach().clone()
    )
    subject_key = "brain_encoder.subject_projection.weights"
    legacy_state[subject_key] = legacy_state[subject_key].transpose(-1, -2).contiguous()
    restored = BrainEmbeddingModel(positions, subject_count=1, config=config)

    restored.load_state_dict(legacy_state, strict=True)

    assert torch.equal(restored.state_dict()[subject_key], current_state[subject_key])


def test_subject_linear_identity_initialization():
    layer = SubjectLinear(
        subject_count=2,
        input_channels=3,
        bias=True,
        initialization="identity",
    )
    features = torch.randn(2, 3, 7)
    output = layer(features, torch.tensor([0, 1]))
    assert torch.equal(output, features)


def test_spatial_channel_merger_supports_subject_specific_2d_positions():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    merger = SpatialChannelMerger(
        positions,
        output_channels=4,
        position_dimension=8,
        subject_count=2,
        per_subject=True,
        dropout_radius=0,
        frequency_start=0,
        fourier_margin=0.2,
        scale_fourier_coordinates=True,
    )
    output = merger(torch.randn(3, 6, 20), torch.tensor([0, 1, 0]))
    assert output.shape == (3, 4, 20)
    assert torch.isfinite(output).all()


def test_spatial_channel_merger_loads_legacy_state_keys_strictly():
    positions = np.stack(
        [np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1
    ).astype("float32")
    source = SpatialChannelMerger(
        positions,
        output_channels=4,
        position_dimension=8,
        subject_count=1,
        per_subject=True,
        frequency_start=0,
        fourier_margin=0.2,
        scale_fourier_coordinates=True,
    )
    legacy_state = {
        "heads": source.coefficients.detach().clone(),
        "channel_positions": source.channel_positions.detach().clone(),
    }
    restored = SpatialChannelMerger(
        positions,
        output_channels=4,
        position_dimension=8,
        subject_count=1,
        per_subject=True,
        frequency_start=0,
        fourier_margin=0.2,
        scale_fourier_coordinates=True,
    )

    restored.load_state_dict(legacy_state, strict=True)

    frequencies_y = torch.arange(2, dtype=restored.channel_positions.dtype)
    frequencies_x = frequencies_y[:, None]
    shifted = restored.channel_positions + 0.2
    phase = (
        shifted[..., None, None, 0] * (2.0 * math.pi * frequencies_x / 1.4)
        + shifted[..., None, None, 1] * (2.0 * math.pi * frequencies_y / 1.4)
    ).reshape(len(positions), -1)
    expected_basis = torch.cat([phase.cos(), phase.sin()], dim=-1)
    assert torch.equal(restored.coefficients, source.coefficients)
    assert torch.equal(restored.basis, expected_basis)


def test_strided_encoder_supports_3d_scalp_positions():
    positions = np.asarray(
        [
            [-0.5, 0.2, 0.8],
            [0.0, 0.7, 0.7],
            [0.5, 0.2, 0.8],
            [-0.4, -0.4, 0.8],
            [0.4, -0.4, 0.8],
            [0.0, 0.0, 1.0],
        ],
        dtype="float32",
    )
    model = build_strided_conv_encoder(
        channel_count=6,
        channel_positions=positions,
        subject_count=2,
        config={
            "embedding_dimension": 8,
            "virtual_channels": 4,
            "fourier_harmonics": 2,
            "dropout": 0,
        },
    )
    output = model(torch.randn(2, 6, 64), torch.tensor([0, 1]))
    assert output.shape == (2, 8)
    assert torch.allclose(
        torch.linalg.vector_norm(output, dim=1), torch.ones(2), atol=1e-5
    )
