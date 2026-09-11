"""跨数据集复用的文本表示与缓存工具。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


def normalize_word(value: str) -> str:
    """按照现有实验规则，将单词清洗为小写字母数字形式。"""
    return "".join(
        character
        for character in str(value).strip()
        if character.isalnum() or character in {"-", "'"}
    ).lower()


def text_embedding_signature(config) -> dict:
    """返回与现有文本向量缓存完全一致的配置签名。"""
    return {
        "model_name": str(config.get("model_name", "t5-large")),
        "layer_fraction": float(config.get("layer_fraction", 0.5)),
        "token_aggregation": str(config.get("token_aggregation", "mean")),
        "add_special_tokens": False,
        "padding_aggregation": "attention_masked",
    }


def load_text_embedding_cache(path, expected_signature=None) -> dict[str, np.ndarray]:
    """读取现有 NPZ 文本向量缓存并验证配置签名。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文本向量缓存不存在：{path}")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"文本向量缓存缺少元数据：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if expected_signature is not None and metadata.get("signature") != expected_signature:
        raise ValueError("文本向量缓存与当前模型/层配置不一致，请使用新的缓存路径。")
    with np.load(path, allow_pickle=False) as payload:
        words = payload["words"].astype(str).tolist()
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if len(words) != len(embeddings) or len(words) != len(set(words)):
        raise ValueError(f"文本向量缓存索引无效：{path}")
    return {word: embeddings[index] for index, word in enumerate(words)}


def ensure_text_embedding_cache(words, config, path) -> Path:
    """增量缓存所选数据划分需要的 T5 词向量。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = text_embedding_signature(config)
    existing = {}
    if path.exists():
        existing = load_text_embedding_cache(path, expected_signature=signature)
    requested = sorted({normalize_word(word) for word in words if normalize_word(word)})
    missing = [word for word in requested if word not in existing]
    if not missing:
        return path

    try:
        from transformers import AutoModelForTextEncoding, AutoTokenizer
    except ImportError as exc:
        raise ImportError("生成 T5 词向量需要 transformers。") from exc

    model_name = signature["model_name"]
    local_only = bool(config.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side="left",
        local_files_only=local_only,
    )
    model = AutoModelForTextEncoding.from_pretrained(
        model_name, local_files_only=local_only
    )
    requested_device = str(config.get("device", "cpu"))
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model.to(device).eval()
    batch_size = int(config.get("batch_size", 32))
    expected_dimension = int(config.get("embedding_dimension", 1024))

    for start in range(0, len(missing), batch_size):
        batch_words = missing[start : start + batch_size]
        print(
            f"T5 词向量 {min(start + len(batch_words), len(missing))}/{len(missing)}"
        )
        inputs = tokenizer(
            batch_words,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        states = outputs.hidden_states
        layer_index = int(signature["layer_fraction"] * len(states) - 1e-6)
        layer_index = min(max(layer_index, 0), len(states) - 1)
        hidden = states[layer_index]
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        if signature["token_aggregation"] == "mean":
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        elif signature["token_aggregation"] == "sum":
            pooled = (hidden * mask).sum(dim=1)
        elif signature["token_aggregation"] == "first":
            pooled = hidden[:, 0]
        elif signature["token_aggregation"] == "last":
            last = inputs["attention_mask"].sum(dim=1).sub(1).clamp_min(0)
            pooled = hidden[torch.arange(len(hidden), device=device), last]
        else:
            raise ValueError(
                f"未知 token aggregation：{signature['token_aggregation']}"
            )
        pooled = pooled.float().cpu().numpy()
        if pooled.shape[1] != expected_dimension:
            raise ValueError(
                f"文本向量维数为 {pooled.shape[1]}，配置预期 {expected_dimension}。"
            )
        for word, embedding in zip(batch_words, pooled):
            existing[word] = embedding.astype(np.float32, copy=False)

    ordered_words = sorted(existing)
    matrix = np.stack([existing[word] for word in ordered_words]).astype(np.float32)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez(file, words=np.asarray(ordered_words), embeddings=matrix)
    os.replace(temporary_path, path)
    metadata = {
        "status": "materialized",
        "signature": signature,
        "word_count": len(ordered_words),
        "embedding_dimension": int(matrix.shape[1]),
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
