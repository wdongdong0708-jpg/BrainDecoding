"""供不同任务共享的可复用神经信号模型组件。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "project_scalp_positions",
    "FourierPositionEmbedding",
    "SpatialChannelMerger",
    "SubjectLinear",
    "StridedConvEncoder",
    "TemporalAttentionPool",
    "DilatedConvEncoder",
    "GroupedTransformerEncoder",
    "BrainEmbeddingModel",
    "build_strided_conv_encoder",
    "build_brain_embedding_model",
]


def project_scalp_positions(channel_positions, margin=0.1):
    """将三维传感器坐标投影到归一化二维平面。"""
    positions = torch.as_tensor(channel_positions, dtype=torch.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("channel_positions must have shape [channels, 3].")
    if not torch.isfinite(positions).all():
        raise ValueError("channel_positions contains non-finite values.")
    if not 0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5).")

    unit = positions / positions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    radius = torch.acos(unit[:, 2].clamp(-1.0, 1.0))
    azimuth = torch.atan2(unit[:, 1], unit[:, 0])
    projected = torch.stack(
        [radius * torch.cos(azimuth), radius * torch.sin(azimuth)], dim=1
    )
    lower = projected.amin(dim=0, keepdim=True)
    span = (projected.amax(dim=0, keepdim=True) - lower).clamp_min(1e-8)
    return margin + (1.0 - 2.0 * margin) * (projected - lower) / span


class FourierPositionEmbedding(nn.Module):
    """使用傅里叶基编码归一化二维传感器位置。"""

    def __init__(
        self,
        dimension=2048,
        frequency_start=0,
        margin=0.2,
        scale_coordinates=True,
    ):
        super().__init__()
        dimension = int(dimension)
        frequency_count = int(round((dimension // 2) ** 0.5))
        if frequency_count * frequency_count * 2 != dimension:
            raise ValueError("dimension must be twice a square number.")
        if int(frequency_start) < 0:
            raise ValueError("frequency_start must be non-negative.")
        if float(margin) < 0:
            raise ValueError("margin must be non-negative.")
        self.dimension = dimension
        self.frequency_count = frequency_count
        self.frequency_start = int(frequency_start)
        self.margin = float(margin)
        self.scale_coordinates = bool(scale_coordinates)

    def forward(self, positions):
        if positions.shape[-1] != 2:
            raise ValueError("positions must contain two-dimensional coordinates.")
        coordinates = positions
        if self.scale_coordinates:
            coordinates = (coordinates + self.margin) / (1.0 + 2.0 * self.margin)
        frequencies = torch.arange(
            self.frequency_start,
            self.frequency_start + self.frequency_count,
            device=positions.device,
            dtype=positions.dtype,
        )
        frequency_x, frequency_y = torch.meshgrid(
            frequencies, frequencies, indexing="ij"
        )
        phase = 2.0 * math.pi * (
            coordinates[..., :1] * frequency_x.flatten()
            + coordinates[..., 1:] * frequency_y.flatten()
        )
        return torch.cat([phase.cos(), phase.sin()], dim=-1)


class SpatialChannelMerger(nn.Module):
    """学习从传感器到虚拟通道的傅里叶位置注意力映射。

    同一组件既支持全局注意力映射，也支持每名被试各用一张映射。
    传感器位置可以是归一化二维坐标，也可以是三维头皮坐标。
    """

    def __init__(
        self,
        channel_positions,
        output_channels=64,
        position_dimension=128,
        subject_count=1,
        per_subject=False,
        dropout_radius=0.0,
        projection_margin=0.1,
        frequency_start=1,
        fourier_margin=0.0,
        scale_fourier_coordinates=False,
        initialization_std=None,
    ):
        super().__init__()
        positions = torch.as_tensor(channel_positions, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError("channel_positions must have shape [channels, 2 or 3].")
        if positions.shape[1] == 3:
            positions = project_scalp_positions(positions, margin=projection_margin)
        elif not torch.isfinite(positions).all():
            raise ValueError("channel_positions contains non-finite values.")
        if int(output_channels) <= 0 or int(subject_count) <= 0:
            raise ValueError("output_channels and subject_count must be positive.")
        if float(dropout_radius) < 0:
            raise ValueError("dropout_radius must be non-negative.")

        self.register_buffer("channel_positions", positions, persistent=False)
        embedding = FourierPositionEmbedding(
            dimension=position_dimension,
            frequency_start=frequency_start,
            margin=fourier_margin,
            scale_coordinates=scale_fourier_coordinates,
        )
        self.position_embedding = embedding
        self.register_buffer("basis", embedding(positions))
        self.per_subject = bool(per_subject)
        self.dropout_radius = float(dropout_radius)

        coefficient_shape = (int(output_channels), int(position_dimension))
        if self.per_subject:
            coefficient_shape = (int(subject_count), *coefficient_shape)
        self.coefficients = nn.Parameter(torch.empty(coefficient_shape))
        standard_deviation = (
            1.0 / math.sqrt(int(position_dimension))
            if initialization_std is None
            else float(initialization_std)
        )
        nn.init.normal_(self.coefficients, mean=0.0, std=standard_deviation)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """兼容组件通用化前保存的 ``heads/channel_positions`` 权重键。"""
        legacy_coefficients_key = prefix + "heads"
        coefficients_key = prefix + "coefficients"
        legacy_positions_key = prefix + "channel_positions"
        basis_key = prefix + "basis"

        if (
            legacy_coefficients_key in state_dict
            and coefficients_key not in state_dict
        ):
            state_dict[coefficients_key] = state_dict.pop(legacy_coefficients_key)

        if legacy_positions_key in state_dict and basis_key not in state_dict:
            legacy_positions = state_dict.pop(legacy_positions_key)
            current_positions = self.channel_positions.to(
                device=legacy_positions.device,
                dtype=legacy_positions.dtype,
            )
            if (
                legacy_positions.shape != current_positions.shape
                or not torch.allclose(legacy_positions, current_positions)
            ):
                error_msgs.append(
                    f"{legacy_positions_key} does not match the current dataset "
                    "channel positions."
                )
            if (
                self.position_embedding.frequency_start != 0
                or not self.position_embedding.scale_coordinates
            ):
                error_msgs.append(
                    f"{legacy_positions_key} uses an unsupported legacy Fourier "
                    "configuration."
                )
            frequency_count = self.position_embedding.frequency_count
            frequencies_y = torch.arange(
                frequency_count,
                device=legacy_positions.device,
                dtype=legacy_positions.dtype,
            )
            frequencies_x = frequencies_y[:, None]
            margin = self.position_embedding.margin
            width = 1.0 + 2.0 * margin
            shifted = legacy_positions + margin
            phase_x = 2.0 * math.pi * frequencies_x / width
            phase_y = 2.0 * math.pi * frequencies_y / width
            shifted = shifted[..., None, None, :]
            phase = (
                shifted[..., 0] * phase_x + shifted[..., 1] * phase_y
            ).reshape(*legacy_positions.shape[:-1], -1)
            state_dict[basis_key] = torch.cat(
                [phase.cos(), phase.sin()], dim=-1
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, signals, subject_indices=None):
        batch_size, channel_count, _ = signals.shape
        if channel_count != len(self.channel_positions):
            raise ValueError("Signal channels and channel_positions do not match.")

        if self.per_subject:
            if subject_indices is None:
                subject_indices = torch.zeros(
                    batch_size, device=signals.device, dtype=torch.long
                )
            subject_indices = subject_indices.long()
            if len(subject_indices) != batch_size:
                raise ValueError("subject_indices and batch size do not match.")
            if subject_indices.min() < 0 or subject_indices.max() >= len(
                self.coefficients
            ):
                raise IndexError("subject index is outside the spatial merger range.")
            coefficients = self.coefficients[subject_indices]
            scores = torch.einsum("bod,cd->boc", coefficients, self.basis)
        else:
            scores = torch.einsum("od,cd->oc", self.coefficients, self.basis)
            scores = scores.unsqueeze(0).expand(batch_size, -1, -1)

        if self.training and self.dropout_radius > 0:
            center = torch.rand(2, device=signals.device, dtype=signals.dtype)
            banned = (
                self.channel_positions.to(dtype=signals.dtype) - center
            ).norm(dim=-1) <= self.dropout_radius
            scores = scores.masked_fill(banned[None, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bct,boc->bot", signals, weights)


class SubjectLinear(nn.Module):
    """为每名已知被试应用独立的线性通道变换。"""

    def __init__(
        self,
        subject_count,
        input_channels,
        output_channels=None,
        bias=True,
        initialization="identity",
    ):
        super().__init__()
        subject_count = int(subject_count)
        input_channels = int(input_channels)
        output_channels = (
            input_channels if output_channels is None else int(output_channels)
        )
        if subject_count <= 0 or input_channels <= 0 or output_channels <= 0:
            raise ValueError("subject_count and channel dimensions must be positive.")
        self.weights = nn.Parameter(
            torch.empty(subject_count, output_channels, input_channels)
        )
        if initialization == "identity":
            if input_channels != output_channels:
                raise ValueError("identity initialization requires equal dimensions.")
            with torch.no_grad():
                self.weights.copy_(torch.eye(input_channels).repeat(subject_count, 1, 1))
        elif initialization == "normal":
            nn.init.normal_(
                self.weights, mean=0.0, std=1.0 / math.sqrt(input_channels)
            )
        else:
            raise ValueError("initialization must be 'identity' or 'normal'.")
        if bias:
            self.bias = nn.Parameter(torch.zeros(subject_count, output_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, features, subject_indices=None):
        if subject_indices is None:
            subject_indices = torch.zeros(
                len(features), device=features.device, dtype=torch.long
            )
        subject_indices = subject_indices.long()
        if len(subject_indices) != len(features):
            raise ValueError("subject_indices and batch size do not match.")
        if subject_indices.min() < 0 or subject_indices.max() >= len(self.weights):
            raise IndexError("subject index is outside the subject layer range.")
        output = torch.bmm(self.weights[subject_indices], features)
        if self.bias is not None:
            output = output + self.bias[subject_indices].unsqueeze(-1)
        return output


class StridedConvEncoder(nn.Module):
    """将固定长度的多通道信号映射为归一化向量。"""

    def __init__(
        self,
        channel_positions,
        subject_count,
        embedding_dimension=768,
        virtual_channels=64,
        fourier_harmonics=8,
        spatial_margin=0.1,
        dropout=0.1,
    ):
        super().__init__()
        self.spatial_attention = SpatialChannelMerger(
            channel_positions,
            output_channels=virtual_channels,
            position_dimension=2 * int(fourier_harmonics) ** 2,
            subject_count=subject_count,
            per_subject=False,
            dropout_radius=0.0,
            projection_margin=spatial_margin,
            frequency_start=1,
            fourier_margin=0.0,
            scale_fourier_coordinates=False,
            initialization_std=0.02,
        )
        self.subject_layer = SubjectLinear(
            subject_count,
            virtual_channels,
            bias=True,
            initialization="identity",
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(virtual_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dimension),
        )

    def forward(self, signals, subject_indices):
        features = self.spatial_attention(signals)
        features = self.subject_layer(features, subject_indices)
        features = self.encoder(features)
        return F.normalize(self.projection(features), dim=-1)


def build_strided_conv_encoder(channel_count, channel_positions, subject_count, config):
    """构建紧凑的跨步卷积信号编码器。"""
    if len(channel_positions) != int(channel_count):
        raise ValueError("Signal channels and channel_positions do not match.")
    return StridedConvEncoder(
        channel_positions=channel_positions,
        subject_count=subject_count,
        embedding_dimension=int(config.get("embedding_dimension", 768)),
        virtual_channels=int(config.get("virtual_channels", 64)),
        fourier_harmonics=int(config.get("fourier_harmonics", 8)),
        spatial_margin=float(config.get("spatial_margin", 0.1)),
        dropout=float(config.get("dropout", 0.1)),
    )


class TemporalAttentionPool(nn.Module):
    """使用学习得到的加性注意力汇聚时间特征序列。"""

    def __init__(self, input_dimension, hidden_dimension=256):
        super().__init__()
        self.key = nn.Linear(int(input_dimension), int(hidden_dimension))
        self.score = nn.Linear(int(hidden_dimension), 1)

    def forward(self, features):
        keys = features.transpose(1, 2)
        scores = self.score(torch.tanh(self.key(keys))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)
        return torch.bmm(weights, keys).squeeze(1)


class DilatedConvEncoder(nn.Module):
    """由空间合并、残差扩张卷积和时间汇聚组成的信号编码器。"""

    def __init__(self, channel_positions, subject_count, output_dimension, config):
        super().__init__()
        virtual_channels = int(config.get("merger_channels", 270))
        position_dimension = int(config.get("merger_position_dimension", 2048))
        initial_channels = int(config.get("initial_linear", 512))
        hidden_channels = int(config.get("hidden", 160))
        depth = int(config.get("depth", 5))
        kernel_size = int(config.get("kernel_size", 3))
        dilation_growth = int(config.get("dilation_growth", 2))
        dilation_period = config.get("dilation_period", 5)
        dilation_period = None if dilation_period is None else int(dilation_period)
        dropout_input = float(config.get("dropout_input", 0.1))
        convolution_dropout = float(config.get("convolution_dropout", 0.0))
        use_batch_norm = bool(config.get("batch_norm", True))
        use_gelu = bool(config.get("gelu", True))
        use_skip = bool(config.get("skip", True))
        glu_every = int(config.get("glu_every", 2))
        glu_context = int(config.get("glu_context", 1))
        if depth <= 0:
            raise ValueError("depth must be positive.")

        self.channel_merger = SpatialChannelMerger(
            channel_positions,
            output_channels=virtual_channels,
            position_dimension=position_dimension,
            subject_count=subject_count,
            per_subject=bool(config.get("merger_per_subject", True)),
            dropout_radius=float(config.get("merger_dropout", 0.2)),
            frequency_start=0,
            fourier_margin=0.2,
            scale_fourier_coordinates=True,
        )
        self.initial_projection = nn.Conv1d(virtual_channels, initial_channels, 1)
        self.subject_projection = SubjectLinear(
            subject_count,
            initial_channels,
            bias=False,
            initialization="normal",
        )

        sizes = [initial_channels]
        sizes.extend([hidden_channels] * max(depth - 1, 0))
        sizes.append(int(output_dimension))
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.residual = []
        dilation = 1
        activation_type = nn.GELU if use_gelu else nn.ReLU
        for index, (input_channels, next_channels) in enumerate(
            zip(sizes[:-1], sizes[1:])
        ):
            if dilation_period and index % dilation_period == 0:
                dilation = 1
            modules = []
            if index == 0 and dropout_input > 0:
                modules.append(nn.Dropout(dropout_input))
            modules.append(
                nn.Conv1d(
                    input_channels,
                    next_channels,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size // 2) * dilation,
                    dilation=dilation,
                )
            )
            dilation *= dilation_growth
            is_last = index == depth - 1
            if not is_last:
                if use_batch_norm:
                    modules.append(nn.BatchNorm1d(next_channels))
                modules.append(activation_type())
                if convolution_dropout > 0:
                    modules.append(nn.Dropout(convolution_dropout))
            self.blocks.append(nn.Sequential(*modules))
            self.residual.append(use_skip and input_channels == next_channels)
            if glu_every and (index + 1) % glu_every == 0:
                self.gates.append(
                    nn.Sequential(
                        nn.Conv1d(
                            next_channels,
                            2 * next_channels,
                            1 + 2 * glu_context,
                            padding=glu_context,
                        ),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.gates.append(nn.Identity())
        self.temporal_attention = TemporalAttentionPool(
            output_dimension,
            hidden_dimension=int(config.get("temporal_attention_hidden", 256)),
        )

    def forward(self, signals, subject_indices=None):
        features = self.channel_merger(signals, subject_indices)
        features = self.initial_projection(features)
        features = self.subject_projection(features, subject_indices)
        for block, gate, residual in zip(self.blocks, self.gates, self.residual):
            previous = features
            features = block(features)
            if residual:
                features = features + previous
            features = gate(features)
        return self.temporal_attention(features)


class GroupedTransformerEncoder(nn.Module):
    """在每个序列组内独立应用上下文 Transformer。"""

    def __init__(self, dimension, config):
        super().__init__()
        try:
            from x_transformers import Encoder
        except ImportError as exc:
            raise ImportError(
                "The context transformer requires the x-transformers package."
            ) from exc
        self.encoder = Encoder(
            dim=int(dimension),
            depth=int(config.get("depth", 16)),
            heads=int(config.get("heads", 16)),
            attn_dropout=float(config.get("attention_dropout", 0.1)),
            ff_dropout=float(config.get("feedforward_dropout", 0.0)),
            use_scalenorm=bool(config.get("use_scalenorm", True)),
            use_rmsnorm=bool(config.get("use_rmsnorm", False)),
            rotary_pos_emb=bool(config.get("rotary_position_embedding", True)),
            residual_attn=bool(config.get("residual_attention", False)),
            scale_residual=bool(config.get("scale_residual", True)),
        )

    def forward(self, features, group_indices):
        if group_indices is None:
            raise ValueError("group_indices is required by the context transformer.")
        group_values = group_indices.detach().cpu().tolist()
        groups = {}
        for row_index, group in enumerate(group_values):
            groups.setdefault(int(group), []).append(row_index)
        max_length = max(len(indices) for indices in groups.values())
        padded = features.new_zeros((len(groups), max_length, features.shape[-1]))
        mask = torch.zeros(
            len(groups), max_length, device=features.device, dtype=torch.bool
        )
        ordered_indices = []
        for group_index, indices in enumerate(groups.values()):
            index_tensor = torch.tensor(indices, device=features.device)
            padded[group_index, : len(indices)] = features[index_tensor]
            mask[group_index, : len(indices)] = True
            ordered_indices.append(index_tensor)
        encoded = self.encoder(padded, mask=mask)
        output = torch.empty_like(features)
        for group_index, indices in enumerate(ordered_indices):
            output[indices] = encoded[group_index, : len(indices)]
        return output


class BrainEmbeddingModel(nn.Module):
    """带可选分组上下文 Transformer 的脑信号编码器。"""

    def __init__(self, channel_positions, subject_count, config):
        super().__init__()
        output_dimension = int(config.get("embedding_dimension", 1024))
        self.brain_encoder = DilatedConvEncoder(
            channel_positions,
            subject_count=subject_count,
            output_dimension=output_dimension,
            config=config.get("conv", config),
        )
        self.use_transformer = bool(config.get("use_transformer", True))
        self.context_mode = str(config.get("context_mode", "grouped"))
        if self.context_mode not in {"grouped", "singleton"}:
            raise ValueError(
                "model.context_mode must be either 'grouped' or 'singleton'."
            )
        self.context_transformer = (
            GroupedTransformerEncoder(
                output_dimension, config.get("transformer", {})
            )
            if self.use_transformer
            else None
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_merger_key = prefix + "brain_encoder.channel_merger.heads"
        subject_weights_key = prefix + "brain_encoder.subject_projection.weights"
        if legacy_merger_key in state_dict and subject_weights_key in state_dict:
            legacy_weights = state_dict[subject_weights_key]
            if legacy_weights.ndim != 3 or legacy_weights.shape[-2] != legacy_weights.shape[-1]:
                error_msgs.append(
                    f"{subject_weights_key} cannot be migrated from the legacy "
                    "feature-times-weight projection."
                )
            else:
                state_dict[subject_weights_key] = legacy_weights.transpose(-1, -2).contiguous()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        signals,
        subject_indices=None,
        group_indices=None,
        return_brain_embedding=False,
    ):
        brain_embedding = F.normalize(
            self.brain_encoder(signals, subject_indices), dim=-1
        )
        output = brain_embedding
        if self.context_transformer is not None:
            if self.context_mode == "singleton":
                group_indices = torch.arange(
                    len(output), device=output.device, dtype=torch.long
                )
            output = self.context_transformer(output, group_indices)
            output = F.normalize(output, dim=-1)
        if return_brain_embedding:
            return output, brain_embedding
        return output


def build_brain_embedding_model(
    channel_count,
    channel_positions,
    subject_count,
    config,
):
    """构建带可选分组上下文的扩张卷积模型。"""
    if len(channel_positions) != int(channel_count):
        raise ValueError("Signal channels and channel_positions do not match.")
    return BrainEmbeddingModel(
        channel_positions=channel_positions,
        subject_count=int(subject_count),
        config=config,
    )
