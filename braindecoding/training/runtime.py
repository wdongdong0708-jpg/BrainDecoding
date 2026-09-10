"""与具体数据集无关的训练运行时工具。"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """同步设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    """按现有设备策略选择 PyTorch 设备。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前环境不可用。")
    return device


def save_json(path, payload):
    """以 UTF-8 格式保存可复核的 JSON 结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def limit_rows(table, maximum):
    """保持原顺序截取小规模调试数据。"""
    if maximum is None or len(table) <= int(maximum):
        return table.reset_index(drop=True)
    return table.iloc[: int(maximum)].reset_index(drop=True)


def cpu_state_dict(module):
    """复制一份与原模块解耦的 CPU 状态字典。"""
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def save_checkpoint(path, payload):
    """通过临时文件原子替换保存检查点。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_checkpoint(path, map_location="cpu"):
    """使用 PyTorch 受限加载器读取仅含状态的检查点格式。"""
    return torch.load(path, map_location=map_location, weights_only=True)


def parameter_count(module):
    """返回模块的参数总数。"""
    return int(sum(parameter.numel() for parameter in module.parameters()))
