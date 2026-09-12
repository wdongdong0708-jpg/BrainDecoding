"""LibriBrain100 Sherlock1 词检索训练入口。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "LibriBrain100.yaml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from braindecoding.config import load_yaml_with_extends, project_path
from braindecoding.experiment import (
    initialize_run_directory,
    resolve_experiment_config,
    update_run_status,
)
from braindecoding.results import build_training_summary
from braindecoding.training.runtime import (
    choose_device,
    cpu_state_dict,
    limit_rows,
    load_checkpoint,
    parameter_count,
    save_checkpoint,
    save_json,
    set_seed,
)
from braindecoding.training.word import (
    SentenceBatchSampler,
    encode_loader,
    make_loader,
    move_batch,
    train_one_epoch,
)
from braindecoding.data import libribrain as dataset_module
from losses import build_siglip_loss
from metrics import fixed_vocabulary_retrieval_metrics
from models import build_brain_embedding_model
from optimizers import (
    build_adamw_for_modules,
    build_cosine_annealing_scheduler,
)


def load_config(path=None):
    config_path = project_path(path or DEFAULT_CONFIG)
    config = load_yaml_with_extends(config_path)
    config = resolve_experiment_config(config)
    for key in ("event_table", "meg_dir", "text_embeddings"):
        config["cache"][key] = str(project_path(config["cache"][key]))
    config["training"]["output_dir"] = str(
        project_path(config["training"]["output_dir"])
    )
    pretrained_checkpoint = config["training"].get(
        "pretrained_brain_encoder_checkpoint"
    )
    if pretrained_checkpoint:
        config["training"]["pretrained_brain_encoder_checkpoint"] = str(
            project_path(pretrained_checkpoint)
        )
    layout_path = config["dataset"].get("layout_path")
    if layout_path:
        config["dataset"]["layout_path"] = str(project_path(layout_path))
    return config


def prepare_event_table(
    config,
    with_meg_cache=False,
    with_text_embeddings=False,
    include_test=False,
    force_cache=False,
):
    """构建并审计事件表，并按需物化所选输入。"""
    event_table = dataset_module.build_event_table(config["dataset"])
    audit = dataset_module.audit_event_table(event_table, config["dataset"])
    event_path = dataset_module.save_event_table(
        event_table, config["cache"]["event_table"], audit=audit
    )
    selected_splits = ["train", "val"] + (["test"] if include_test else [])
    selected = event_table[
        event_table["split"].isin(selected_splits)
        & event_table["is_trainable"].astype(bool)
    ].reset_index(drop=True)
    if with_text_embeddings:
        dataset_module.ensure_text_embedding_cache(
            selected["normalized_word"],
            config["text_embedding"],
            config["cache"]["text_embeddings"],
        )
    if with_meg_cache:
        dataset_module.ensure_recording_caches(
            selected,
            config["dataset"],
            config["cache"]["meg_dir"],
            force=force_cache,
        )
    return {
        "event_table": str(event_path),
        "audit": audit,
        "materialized_splits": selected_splits,
        "text_embeddings_materialized": bool(with_text_embeddings),
        "meg_materialized": bool(with_meg_cache),
    }


def ensure_event_table(config):
    event_path = Path(config["cache"]["event_table"])
    if not event_path.exists():
        prepare_event_table(config)
    return event_path


def materialize_selected_inputs(config, selected_table, force_cache=False):
    dataset_module.ensure_text_embedding_cache(
        selected_table["normalized_word"],
        config["text_embedding"],
        config["cache"]["text_embeddings"],
    )
    dataset_module.ensure_recording_caches(
        selected_table,
        config["dataset"],
        config["cache"]["meg_dir"],
        force=force_cache,
    )


def build_dataset(config, table, zero_meg=False):
    return dataset_module.LibriBrainWordDataset(
        table,
        config["dataset"],
        config["cache"]["meg_dir"],
        config["cache"]["text_embeddings"],
        config["text_embedding"],
        zero_meg=zero_meg,
    )


def set_brain_encoder_trainable(model, trainable):
    """冻结时同时固定脑信号编码器的随机层和归一化统计。"""
    trainable = bool(trainable)
    for parameter in model.brain_encoder.parameters():
        parameter.requires_grad_(trainable)
    model.brain_encoder.train(trainable)


def _file_sha256(path):
    """计算检查点摘要，固定预热来源。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pretrained_brain_encoder(
    model,
    checkpoint_path,
    config,
    channel_names,
    channel_positions,
):
    """仅载入同一 LibriBrain100 合同下 CNN-only 检查点的脑信号编码器。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "word_decoding/LibriBrain100":
        raise ValueError(f"预热来源不是 LibriBrain100 检查点：{checkpoint_path}")
    source_model_config = checkpoint.get("model_config", {})
    if bool(source_model_config.get("use_transformer", True)):
        raise ValueError("预热来源必须是 CNN-only 检查点。")
    if source_model_config.get("embedding_dimension") != config["model"].get(
        "embedding_dimension"
    ):
        raise ValueError("预热 CNN 与目标模型的输出维度不同。")
    if source_model_config.get("conv") != config["model"].get("conv"):
        raise ValueError("预热 CNN 与目标模型的卷积结构不同。")

    source_dataset = checkpoint.get("dataset_contract", {})
    for field in ("root", "task", "split", "window_seconds", "target_sampling_rate_hz"):
        if source_dataset.get(field) != config["dataset"].get(field):
            raise ValueError(f"预热 CNN 的数据合同在 {field} 上不同。")
    if tuple(checkpoint.get("channel_names", ())) != tuple(channel_names):
        raise ValueError("预热 CNN 与当前数据的通道顺序不同。")
    source_positions = np.asarray(checkpoint.get("channel_positions", ()))
    target_positions = np.asarray(channel_positions)
    if source_positions.shape != target_positions.shape or not np.allclose(
        source_positions, target_positions, rtol=0.0, atol=1e-7
    ):
        raise ValueError("预热 CNN 与当前数据的通道位置不同。")

    prefix = "brain_encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError("预热检查点不包含 brain_encoder 参数。")
    model.brain_encoder.load_state_dict(encoder_state, strict=True)
    return {
        "method": "cnn_only_best_checkpoint_brain_encoder_only",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _file_sha256(checkpoint_path),
        "source_epoch": int(checkpoint["epoch"]),
        "source_macro_r_at_10": checkpoint.get("metrics", {}).get(
            "retrieval_acc10_vocab=libribrain50_macro"
        ),
        "loaded_loss_state": False,
        "loaded_transformer_state": False,
    }


def evaluate_loader(model, loader, device, top_ks=(1, 10), amp=True):
    encoded = encode_loader(model, loader, device, amp=amp)
    metrics = fixed_vocabulary_retrieval_metrics(
        encoded["predictions"],
        encoded["targets"],
        encoded["words"],
        dataset_module.LIBRIBRAIN100_50_WORD_VOCABULARY,
        top_ks=top_ks,
        vocabulary_name="libribrain50",
    )
    return metrics, encoded


def checkpoint_payload(
    model,
    loss_module,
    config,
    epoch,
    metrics,
    channel_names,
    channel_positions,
    optimizer=None,
    scheduler=None,
    initialization_audit=None,
    training_phase=None,
):
    payload = {
        "format_version": 1,
        "task": "word_decoding/LibriBrain100",
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "loss_state": loss_module.state_dict(),
        "model_config": config["model"],
        "loss_config": config["loss"],
        "metrics": metrics,
        "channel_names": list(channel_names),
        "channel_positions": np.asarray(channel_positions).tolist(),
        "dataset_contract": {
            "root": config["dataset"]["root"],
            "task": config["dataset"]["task"],
            "split": config["dataset"]["split"],
            "window_seconds": config["dataset"]["window_seconds"],
            "target_sampling_rate_hz": config["dataset"][
                "target_sampling_rate_hz"
            ],
        },
        "provenance": config.get("provenance", {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if initialization_audit is not None:
        payload["initialization_audit"] = initialization_audit
    if training_phase is not None:
        payload["training_phase"] = training_phase
    return payload


def run_training(config, smoke=False, save=True, force_cache=False):
    training_config = dict(config["training"])
    staged_training = bool(
        training_config.get("pretrained_brain_encoder_checkpoint")
    )
    if smoke:
        training_config["epochs"] = 2 if staged_training else 1
        training_config["patience"] = training_config["epochs"]
        training_config["batch_size"] = min(
            int(training_config["batch_size"]), 16
        )
        if staged_training:
            training_config["freeze_brain_encoder_epochs"] = 1
    seed = int(training_config["seed"])
    set_seed(seed)
    device = choose_device(training_config.get("device", "auto"))
    event_path = ensure_event_table(config)
    subjects = config["dataset"].get("subjects")
    train_table = dataset_module.load_event_table(
        event_path, split="train", subjects=subjects
    )
    val_table = dataset_module.load_event_table(
        event_path, split="val", subjects=subjects
    )
    if smoke:
        maximum = int(training_config.get("smoke_rows", 64))
        train_table = limit_rows(train_table, maximum)
        val_table = limit_rows(val_table, maximum)

    materialization_table = pd.concat(
        [train_table, val_table], ignore_index=True
    )
    materialize_selected_inputs(
        config, materialization_table, force_cache=force_cache
    )
    train_dataset = build_dataset(config, train_table)
    val_dataset = build_dataset(config, val_table)
    train_loader, train_sampler = make_loader(
        train_dataset, training_config, shuffle=True
    )
    val_loader, _ = make_loader(val_dataset, training_config, shuffle=False)

    model = build_brain_embedding_model(
        train_dataset.channel_count,
        train_dataset.channel_positions,
        train_dataset.subject_count,
        config["model"],
    ).to(device)
    loss_module = build_siglip_loss(config["loss"]).to(device)
    initialization_audit = None
    if staged_training:
        if model.context_transformer is None:
            raise ValueError("CNN 预热要求目标模型启用 Transformer。")
        initialization_audit = load_pretrained_brain_encoder(
            model,
            training_config["pretrained_brain_encoder_checkpoint"],
            config,
            train_dataset.channel_names,
            train_dataset.channel_positions,
        )
    optimizer = build_adamw_for_modules([model, loss_module], training_config)
    freeze_brain_encoder_epochs = int(
        training_config.get("freeze_brain_encoder_epochs", 0)
    )
    if freeze_brain_encoder_epochs < 0:
        raise ValueError("freeze_brain_encoder_epochs 不能为负数。")
    if staged_training and not 0 < freeze_brain_encoder_epochs < int(
        training_config["epochs"]
    ):
        raise ValueError("CNN 预热的冻结轮数必须大于零且小于总轮数。")
    if not staged_training and freeze_brain_encoder_epochs:
        raise ValueError("未提供预热检查点时不能冻结 brain_encoder。")
    scheduler = build_cosine_annealing_scheduler(
        optimizer,
        int(training_config["epochs"]),
        minimum_learning_rate=float(
            training_config.get("minimum_learning_rate", 0.0)
        ),
    )
    use_amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    top_ks = tuple(int(value) for value in training_config.get("top_ks", [1, 10]))
    selection_metric = f"retrieval_acc{max(top_ks)}_vocab=libribrain50_macro"
    output_dir = Path(training_config["output_dir"])
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_epoch = None
    best_metrics = None
    best_model_state = None
    best_loss_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(int(training_config["epochs"])):
        train_sampler.set_epoch(epoch)
        encoder_is_frozen = epoch < freeze_brain_encoder_epochs
        set_brain_encoder_trainable(model, not encoder_is_frozen)
        train_loss = train_one_epoch(
            model,
            loss_module,
            train_loader,
            optimizer,
            scaler,
            device,
            training_config,
            freeze_brain_encoder=encoder_is_frozen,
        )
        validation_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            top_ks=top_ks,
            amp=training_config.get("amp", True),
        )
        scheduler.step()
        score = float(validation_metrics[selection_metric])
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **validation_metrics,
        }
        history.append(epoch_record)
        print(
            f"第 {epoch + 1} 轮：loss={train_loss:.5f}，"
            f"val macro Top-{max(top_ks)}={score:.6f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = validation_metrics
            epochs_without_improvement = 0
            if save:
                save_checkpoint(
                    output_dir / "best.pt",
                    checkpoint_payload(
                        model,
                        loss_module,
                        config,
                        epoch + 1,
                        validation_metrics,
                        train_dataset.channel_names,
                        train_dataset.channel_positions,
                        initialization_audit=initialization_audit,
                        training_phase={
                            "freeze_brain_encoder_epochs": freeze_brain_encoder_epochs,
                            "encoder_frozen_this_epoch": encoder_is_frozen,
                        },
                    ),
                )
            else:
                best_model_state = cpu_state_dict(model)
                best_loss_state = cpu_state_dict(loss_module)
        else:
            epochs_without_improvement += 1

        if save:
            save_checkpoint(
                output_dir / "last.pt",
                checkpoint_payload(
                    model,
                    loss_module,
                    config,
                    epoch + 1,
                    validation_metrics,
                    train_dataset.channel_names,
                    train_dataset.channel_positions,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    initialization_audit=initialization_audit,
                    training_phase={
                        "freeze_brain_encoder_epochs": freeze_brain_encoder_epochs,
                        "encoder_frozen_this_epoch": encoder_is_frozen,
                    },
                ),
            )
        if epochs_without_improvement >= int(training_config["patience"]):
            break

    if best_epoch is None:
        raise RuntimeError("训练没有产生有效的验证指标。")
    if not save and best_model_state is not None:
        model.load_state_dict(best_model_state)
        loss_module.load_state_dict(best_loss_state)

    summary = {
        "status": "smoke_completed" if smoke else "training_completed",
        "device": str(device),
        "test_eeg_opened": False,
        "event_table": str(event_path),
        "train_rows": len(train_dataset),
        "validation_rows": len(val_dataset),
        "meg_shape": [train_dataset.channel_count, train_dataset.window_samples],
        "model_parameter_count": parameter_count(model),
        "loss_parameter_count": parameter_count(loss_module),
        "initialization_audit": initialization_audit,
        "freeze_brain_encoder_epochs": freeze_brain_encoder_epochs,
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_validation": best_metrics,
        "history": history,
        "test_status": "locked_not_evaluated",
    }
    if save:
        saved_summary = (
            build_training_summary(config, summary)
            if "experiment" in config
            else summary
        )
        save_json(output_dir / "training_summary.json", saved_summary)
    return summary


def print_training_summary(summary):
    print("LibriBrain100 训练完成")
    print(f"  device: {summary['device']}")
    print(
        f"  train/val: {summary['train_rows']} / {summary['validation_rows']}"
    )
    print(
        f"  best epoch: {summary['best_epoch']}，"
        f"{summary['selection_metric']}: {summary['best_score']:.6f}"
    )
    print("  test: locked_not_evaluated")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--prepare", action="store_true", help="构建并审计事件表")
    parser.add_argument(
        "--with-meg-cache",
        action="store_true",
        help="prepare 时同时物化 train/val MEG 预处理缓存",
    )
    parser.add_argument(
        "--with-text-embeddings",
        action="store_true",
        help="prepare 时同时物化 train/val T5 词向量",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="仅在显式请求时把 test 加入物化范围",
    )
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    if args.prepare:
        result = prepare_event_table(
            config,
            with_meg_cache=args.with_meg_cache,
            with_text_embeddings=args.with_text_embeddings,
            include_test=args.include_test,
            force_cache=args.force_cache,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    run_output = None
    if "experiment" in config and not args.no_save:
        command_args = list(argv) if argv is not None else sys.argv[1:]
        run_output, _ = initialize_run_directory(
            config,
            [sys.executable, str(Path(__file__).resolve()), *command_args],
        )
    try:
        summary = run_training(
            config,
            smoke=args.smoke,
            save=not args.no_save,
            force_cache=args.force_cache,
        )
    except BaseException:
        if run_output is not None:
            update_run_status(run_output, "failed")
        raise
    if run_output is not None:
        update_run_status(run_output, "completed")
    print_training_summary(summary)


if __name__ == "__main__":
    main()
