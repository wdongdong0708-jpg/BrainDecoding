"""跨任务共享的训练损失。"""

import torch
from torch import nn
from torch.nn import functional as F


def directional_multi_positive_loss(logits, positive_mask):
    """允许一个样本在批次内对应多个同文本正样本。"""
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_infinity)
    numerator = torch.logsumexp(positive_logits, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def multi_positive_contrastive_loss(eeg_embedding, text_embedding, text_keys, temperature=0.07):
    """计算对称的 EEG—文本多正样本对比损失。"""
    eeg_embedding = F.normalize(eeg_embedding, dim=-1)
    text_embedding = F.normalize(text_embedding, dim=-1)
    logits = eeg_embedding @ text_embedding.T / float(temperature)
    positive_mask = text_keys[:, None].eq(text_keys[None, :])
    eeg_loss = directional_multi_positive_loss(logits, positive_mask)
    text_loss = directional_multi_positive_loss(logits.T, positive_mask.T)
    return 0.5 * (eeg_loss + text_loss)


def siglip_contrastive_loss(
    estimate,
    candidate,
    log_temperature,
    bias,
    identical_candidates_threshold=0.999,
):
    """计算 LibriBrain 基线中可识别重复目标的 SigLIP 目标函数。

    文本向量在目标函数内部归一化。脑信号估计应已由模型完成归一化，
    与源实验的 ``norm_kind='y'`` 约定一致。
    """
    if estimate.ndim != 2 or candidate.ndim != 2:
        raise ValueError("SigLIP 输入必须是二维的 batch×feature 张量。")
    if estimate.shape != candidate.shape:
        raise ValueError(
            f"当前任务要求一一配对的估计与目标，实际为 {estimate.shape} / {candidate.shape}。"
        )
    normalized_candidate = F.normalize(candidate, dim=-1, eps=1e-15)
    scores = log_temperature.exp() * (estimate @ normalized_candidate.T) + bias

    if identical_candidates_threshold is None:
        targets = torch.eye(
            len(scores), device=scores.device, dtype=scores.dtype
        )
        weights = None
    else:
        candidate_similarity = normalized_candidate @ normalized_candidate.T
        positives = candidate_similarity >= float(identical_candidates_threshold)
        targets = positives.to(scores.dtype)
        # 重复文本目标属于正样本而非负样本。保留一个对角正样本项，
        # 并屏蔽其余冗余的非对角副本。
        weights = (~positives).to(scores.dtype)
        weights += torch.eye(
            len(scores), device=scores.device, dtype=scores.dtype
        )
    return F.binary_cross_entropy_with_logits(
        scores, targets, weight=weights, reduction="sum"
    ) / max(len(scores), 1)


class SigLipLoss(nn.Module):
    """LibriBrain 词解码使用的可学习温度 SigLIP 损失。"""

    def __init__(
        self,
        initial_temperature=10.0,
        initial_bias=-10.0,
        identical_candidates_threshold=0.999,
    ):
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("SigLIP temperature 必须为正数。")
        self.log_temperature = nn.Parameter(
            torch.tensor(float(initial_temperature)).log()
        )
        self.bias = nn.Parameter(torch.tensor(float(initial_bias)))
        self.identical_candidates_threshold = identical_candidates_threshold

    def forward(self, estimate, candidate):
        return siglip_contrastive_loss(
            estimate,
            candidate,
            self.log_temperature,
            self.bias,
            identical_candidates_threshold=self.identical_candidates_threshold,
        )


def build_siglip_loss(config=None):
    """根据普通配置构建共享的 LibriBrain SigLIP 损失。"""
    config = config or {}
    return SigLipLoss(
        initial_temperature=float(config.get("initial_temperature", 10.0)),
        initial_bias=float(config.get("initial_bias", -10.0)),
        identical_candidates_threshold=config.get(
            "identical_candidates_threshold", 0.999
        ),
    )
