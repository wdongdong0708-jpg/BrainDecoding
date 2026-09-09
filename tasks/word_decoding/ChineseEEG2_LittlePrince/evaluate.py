"""显式评估 ChineseEEG2《小王子》闭集词解码检查点。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from . import train as training
except ImportError:
    import train as training

from datasets import ChineseEEG2 as dataset_module
from models import build_brain_embedding_model


def 验证检查点合同(checkpoint, config, dataset, vocabulary):
    """拒绝数据、通道、被试或候选词合同不一致的检查点。"""
    if checkpoint.get("task") != "word_decoding/ChineseEEG2_LittlePrince":
        raise ValueError("检查点不属于 ChineseEEG2《小王子》任务。")
    if checkpoint.get("text_embedding_config") != config["text_embedding"]:
        raise ValueError("检查点与评估配置的文本向量合同不同。")
    if tuple(checkpoint.get("channel_names", ())) != tuple(dataset.channel_names):
        raise ValueError("检查点与评估数据的脑电通道顺序不同。")
    positions = np.asarray(checkpoint.get("channel_positions", ()))
    if positions.shape != dataset.channel_positions.shape or not np.allclose(
        positions, dataset.channel_positions, rtol=0.0, atol=1e-7
    ):
        raise ValueError("检查点与评估数据的电极坐标不同。")
    if tuple(checkpoint.get("subject_ids", ())) != tuple(dataset.subject_ids):
        raise ValueError("检查点与评估数据的受试者索引合同不同。")
    contract = checkpoint.get("dataset_contract", {})
    if contract.get("context_grouping") != config["dataset"]["context_grouping"]:
        raise ValueError("检查点与评估配置的语境分组方式不同。")
    if tuple(contract.get("evaluation_vocabulary", ())) != tuple(vocabulary):
        raise ValueError("检查点与事件表的训练集候选词不同。")
    for key in (
        "split",
        "window_start_offset_seconds",
        "window_seconds",
        "eligibility_window_seconds",
        "source_sampling_rate_hz",
        "target_sampling_rate_hz",
        "training_vocabulary_policy",
        "evaluation_vocabulary_policy",
    ):
        if contract.get(key) != config["dataset"].get(key):
            raise ValueError(f"检查点的数据合同字段不一致：{key}")


def 评估检查点(
    config,
    checkpoint_path,
    split="val",
    smoke=False,
    zero_eeg_control=False,
    save=True,
    force_cache=False,
):
    """只访问明确指定的数据划分，并可选执行零脑电对照。"""
    if split not in {"val", "test"}:
        raise ValueError("评估划分只能是 val 或 test。")
    training.set_seed(int(config["training"]["seed"]))
    device = training.choose_device(config["training"].get("device", "auto"))
    event_path = training.确保事件表(config)
    table = dataset_module.载入事件表(event_path, split=split)
    if smoke:
        table = training.limit_rows(
            table, int(config["training"].get("smoke_rows", 64))
        )
    training.物化所选输入(config, table, force_cache=force_cache)
    dataset = training.构建数据集(config, table)
    loader, _ = training.make_loader(dataset, config["training"], shuffle=False)
    vocabulary = dataset_module.载入候选词(event_path)

    checkpoint = training.load_checkpoint(checkpoint_path, map_location="cpu")
    验证检查点合同(checkpoint, config, dataset, vocabulary)
    model = build_brain_embedding_model(
        dataset.channel_count,
        dataset.channel_positions,
        dataset.subject_count,
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    top_ks = tuple(int(value) for value in config["training"].get("top_ks", [1, 10]))
    metrics, _ = training.评估数据(
        model,
        loader,
        device,
        vocabulary,
        top_ks=top_ks,
        amp=config["training"].get("amp", True),
    )
    summary = {
        "status": "smoke_evaluation_completed" if smoke else "evaluation_completed",
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "query_rows": len(dataset),
        "test_eeg_opened": split == "test",
        "metrics": metrics,
        "controls": {},
    }
    if zero_eeg_control:
        zero_dataset = training.构建数据集(config, table, zero_eeg=True)
        zero_loader, _ = training.make_loader(
            zero_dataset, config["training"], shuffle=False
        )
        zero_metrics, _ = training.评估数据(
            model,
            zero_loader,
            device,
            vocabulary,
            top_ks=top_ks,
            amp=config["training"].get("amp", True),
        )
        summary["controls"]["zero_eeg"] = zero_metrics

    if save:
        output_dir = Path(config["training"]["output_dir"])
        suffix = "_smoke" if smoke else ""
        training.save_json(
            output_dir / f"evaluation_{split}{suffix}.json", summary
        )
    return summary


def 打印评估摘要(summary):
    """打印主要宏平均指标和候选词覆盖情况。"""
    metrics = summary["metrics"]
    print(f"ChineseEEG2《小王子》{summary['split']} 评估完成")
    print(f"  查询数：{summary['query_rows']}")
    print(
        "  宏平均 Top-10："
        f"{metrics['retrieval_acc10_vocab=chineseeeg2_littleprince50_macro']:.6f}"
    )
    print(
        "  实际出现候选词："
        f"{metrics['observed_vocabulary_size']}/{metrics['vocabulary_size']}"
    )
    if summary["controls"]:
        control = summary["controls"]["zero_eeg"]
        print(
            "  零脑电宏平均 Top-10："
            f"{control['retrieval_acc10_vocab=chineseeeg2_littleprince50_macro']:.6f}"
        )


def 解析参数(argv=None):
    """解析显式验证或测试评估参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(training.DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--zero-eeg-control", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    """执行显式评估命令。"""
    args = 解析参数(argv)
    config = training.载入配置(args.config)
    split = args.split or config.get("evaluation", {}).get("default_split", "val")
    checkpoint = args.checkpoint or str(
        Path(config["training"]["output_dir"]) / "best.pt"
    )
    zero_eeg = args.zero_eeg_control or bool(
        config.get("evaluation", {}).get("zero_eeg_control", False)
    )
    summary = 评估检查点(
        config,
        checkpoint,
        split=split,
        smoke=args.smoke,
        zero_eeg_control=zero_eeg,
        save=not args.no_save,
        force_cache=args.force_cache,
    )
    打印评估摘要(summary)


if __name__ == "__main__":
    main()
