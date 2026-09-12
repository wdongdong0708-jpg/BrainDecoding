"""显式验证或测试 SMN4Lang 检查点。"""

from __future__ import annotations

import argparse
from pathlib import Path


from . import train as training

from braindecoding.config import PROJECT_ROOT
from braindecoding.data import smn4lang as dataset_module
from braindecoding.experiment import evaluation_output_path
from braindecoding.results import (
    canonical_evaluation_from_legacy,
    evaluate_vocabulary_manifests,
    load_vocabulary_assets,
    write_evaluation_result,
)
from braindecoding.models import build_brain_embedding_model
from braindecoding.evaluation.ovmi import fixed_vocabulary_ovmi_metrics


def evaluate_checkpoint(
    config,
    checkpoint_path,
    split="val",
    smoke=False,
    zero_meg_control=False,
    save=True,
    force_cache=False,
):
    """只评估明确指定的一个划分，绝不隐式访问测试集。"""
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be 'val' or 'test'.")
    training.set_seed(int(config["training"]["seed"]))
    device = training.choose_device(config["training"].get("device", "auto"))
    event_path = training.ensure_event_table(config)
    table = dataset_module.load_event_table(
        event_path, split=split, subjects=config["dataset"].get("subjects")
    )
    if smoke:
        table = training.limit_rows(
            table, int(config["training"].get("smoke_rows", 64))
        )
    training.materialize_selected_inputs(config, table, force_cache=force_cache)
    dataset = training.build_dataset(config, table)
    loader, _ = training.make_loader(dataset, config["training"], shuffle=False)

    checkpoint = training.load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "word_decoding/SMN4Lang":
        raise ValueError(f"Not an SMN4Lang checkpoint: {checkpoint_path}")
    checkpoint_text_config = checkpoint.get("text_embedding_config")
    if (
        checkpoint_text_config is not None
        and checkpoint_text_config != config["text_embedding"]
    ):
        raise ValueError("Checkpoint and evaluation text-embedding contracts differ.")
    checkpoint_channels = tuple(checkpoint.get("channel_names", ()))
    if checkpoint_channels and checkpoint_channels != tuple(dataset.channel_names):
        raise ValueError("Checkpoint and dataset MEG channel order differ.")
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
        dataset_module.SMN4LANG50_VOCABULARY,
        config.get("evaluation", {}).get("ovmi", {}),
        base_dir=PROJECT_ROOT,
    )
    summary = {
        "status": "smoke_evaluation_completed" if smoke else "evaluation_completed",
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "query_rows": len(dataset),
        "test_meg_opened": split == "test",
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
        if "experiment" in config:
            vocabulary_manifests, story_reference = load_vocabulary_assets(
                PROJECT_ROOT / "experiments" / "manifests" / "smn4lang"
            )
            vocabulary_results = evaluate_vocabulary_manifests(
                encoded["predictions"],
                encoded["targets"],
                encoded["words"],
                vocabulary_manifests,
                story_reference,
                language="zh",
            )
            canonical = canonical_evaluation_from_legacy(
                config,
                summary,
                event_table_path=event_path,
                subjects=table["subject_id"].astype(str).drop_duplicates().tolist(),
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                vocabulary_results=vocabulary_results,
            )
            write_evaluation_result(config, canonical, smoke=smoke)
        else:
            training.save_json(
                evaluation_output_path(config, split, smoke=smoke), summary
            )
    return summary


def print_evaluation_summary(summary):
    metrics = summary["metrics"]
    print(f"SMN4Lang {summary['split']} evaluation completed")
    print(f"  queries: {summary['query_rows']}")
    print(
        "  macro Top-10: "
        f"{metrics['retrieval_acc10_vocab=smn4lang50_macro']:.6f}"
    )
    print(
        "  observed vocabulary: "
        f"{metrics['observed_vocabulary_size']}/{metrics['vocabulary_size']}"
    )
    if summary["controls"]:
        control = summary["controls"]["zero_meg"]
        print(
            "  zero-MEG macro Top-10: "
            f"{control['retrieval_acc10_vocab=smn4lang50_macro']:.6f}"
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
    split = args.split or config.get("evaluation", {}).get("default_split", "val")
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
