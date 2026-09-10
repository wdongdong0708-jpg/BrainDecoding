"""显式验证或测试 LibriBrain100 检查点。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from . import train as training
except ImportError:  # 通过脚本或 run.py 直接加载时没有父包。
    import train as training

from datasets import LibriBrain as dataset_module
from models import build_brain_embedding_model
from ovmi_metrics import fixed_vocabulary_ovmi_metrics


def evaluate_checkpoint(
    config,
    checkpoint_path,
    split="val",
    smoke=False,
    zero_meg_control=False,
    save=True,
    force_cache=False,
):
    """只评估明确指定的一个划分，绝不隐式评估测试集。"""
    if split not in {"val", "test"}:
        raise ValueError("评估 split 只能是 val 或 test。")
    training.set_seed(int(config["training"]["seed"]))
    device = training.choose_device(config["training"].get("device", "auto"))
    event_path = training.ensure_event_table(config)
    table = dataset_module.load_event_table(event_path, split=split)
    if smoke:
        table = training.limit_rows(
            table, int(config["training"].get("smoke_rows", 64))
        )
    training.materialize_selected_inputs(
        config, table, force_cache=force_cache
    )
    dataset = training.build_dataset(config, table)
    loader, _ = training.make_loader(
        dataset, config["training"], shuffle=False
    )

    checkpoint = training.load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "word_decoding/LibriBrain100":
        raise ValueError(f"不是 LibriBrain100 checkpoint：{checkpoint_path}")
    checkpoint_channels = tuple(checkpoint.get("channel_names", ()))
    if checkpoint_channels and checkpoint_channels != tuple(dataset.channel_names):
        raise ValueError("checkpoint 与当前数据的 MEG 通道顺序不一致。")
    model = build_brain_embedding_model(
        dataset.channel_count,
        dataset.channel_positions,
        dataset.subject_count,
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)

    top_ks = tuple(int(value) for value in config["training"].get("top_ks", [1, 10]))
    metrics, encoded = training.evaluate_loader(
        model,
        loader,
        device,
        top_ks=top_ks,
        amp=config["training"].get("amp", True),
    )
    metrics["ovmi"] = fixed_vocabulary_ovmi_metrics(
        encoded["predictions"],
        encoded["targets"],
        encoded["words"],
        dataset_module.LIBRIBRAIN100_50_WORD_VOCABULARY,
        config.get("evaluation", {}).get("ovmi", {}),
        base_dir=PROJECT_ROOT,
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
    if zero_meg_control:
        zero_dataset = training.build_dataset(config, table, zero_meg=True)
        zero_loader, _ = training.make_loader(
            zero_dataset, config["training"], shuffle=False
        )
        zero_metrics, _ = training.evaluate_loader(
            model,
            zero_loader,
            device,
            top_ks=top_ks,
            amp=config["training"].get("amp", True),
        )
        summary["controls"]["zero_meg"] = zero_metrics

    if save:
        output_dir = Path(config["training"]["output_dir"])
        suffix = "_smoke" if smoke else ""
        training.save_json(
            output_dir / f"evaluation_{split}{suffix}.json", summary
        )
    return summary


def print_evaluation_summary(summary):
    metrics = summary["metrics"]
    print(f"LibriBrain100 {summary['split']} 评估完成")
    print(f"  queries: {summary['query_rows']}")
    print(
        "  macro Top-10: "
        f"{metrics['retrieval_acc10_vocab=libribrain50_macro']:.6f}"
    )
    print(
        "  observed vocabulary: "
        f"{metrics['observed_vocabulary_size']}/{metrics['vocabulary_size']}"
    )
    if summary["controls"]:
        control = summary["controls"]["zero_meg"]
        print(
            "  zero-MEG macro Top-10: "
            f"{control['retrieval_acc10_vocab=libribrain50_macro']:.6f}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(training.DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--zero-meg-control", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = training.load_config(args.config)
    split = args.split or config.get("evaluation", {}).get(
        "default_split", "val"
    )
    checkpoint = args.checkpoint or str(
        Path(config["training"]["output_dir"]) / "best.pt"
    )
    zero_meg = args.zero_meg_control or bool(
        config.get("evaluation", {}).get("zero_meg_control", False)
    )
    summary = evaluate_checkpoint(
        config,
        checkpoint,
        split=split,
        smoke=args.smoke,
        zero_meg_control=zero_meg,
        save=not args.no_save,
        force_cache=args.force_cache,
    )
    print_evaluation_summary(summary)


if __name__ == "__main__":
    main()
