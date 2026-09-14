"""Pallier2025 多受试者词解码训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends, project_path
from braindecoding.data import pallier2025 as dataset_module
from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval_metrics
from braindecoding.experiment import file_sha256, resolve_experiment_config
from braindecoding.losses import build_siglip_loss
from braindecoding.models import build_brain_embedding_model
from braindecoding.optimizers import (
    build_adamw_for_modules,
    build_cosine_annealing_scheduler,
)
from braindecoding.results import build_training_summary, load_vocabulary_assets
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
from braindecoding.training.word import encode_loader, make_loader, move_batch


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "word_decoding"
    / "pallier2025"
    / "sub01-10"
    / "main_word.yaml"
)
VOCABULARY_ASSET_DIRECTORY = (
    PROJECT_ROOT / "experiments" / "manifests" / "pallier2025"
)
SELECTION_VOCABULARY_SIZE = 50
SELECTION_VOCABULARY_NAME = "pallier2025_50"


def load_config(path=None):
    """载入 canonical Pallier 实验配置并解析 derived 路径。"""
    config_path = project_path(path or DEFAULT_CONFIG)
    config = resolve_experiment_config(load_yaml_with_extends(config_path))
    for key in ("event_table", "meg_dir", "text_embeddings"):
        config["cache"][key] = str(project_path(config["cache"][key]))
    for key in ("split_manifest", "qc_artifact", "layout_path"):
        if config["dataset"].get(key):
            config["dataset"][key] = str(project_path(config["dataset"][key]))
    config["training"]["output_dir"] = str(
        project_path(config["training"]["output_dir"])
    )
    return config


def ensure_canonical_inputs(config):
    """只验证冻结数据产品；训练入口不得回退或重新构建旧缓存。"""
    from braindecoding.data.build import check_dataset

    status = check_dataset("pallier2025")
    if status["status"] != "complete":
        raise RuntimeError("Pallier2025 canonical derived data 尚未完整。")
    for key in ("event_table", "meg_dir", "text_embeddings"):
        if not Path(config["cache"][key]).exists():
            raise FileNotFoundError(f"Pallier2025 canonical 数据产品缺失：{key}")
    return Path(config["cache"]["event_table"])


def build_dataset(config, table, zero_meg=False):
    return dataset_module.Pallier2025WordDataset(
        table,
        config["dataset"],
        config["cache"]["meg_dir"],
        config["cache"]["text_embeddings"],
        config["text_embedding"],
        zero_meg=zero_meg,
    )


def loader_group_column(config) -> str:
    """上下文模型使用 recording-local key；word 模型保留原批次组织。"""
    if not bool(config["model"].get("use_transformer", False)):
        return "sentence_uid"
    expected = dataset_module.RUNTIME_CONTEXT_GROUPING
    actual = config["dataset"].get("runtime_context_grouping")
    if actual != expected:
        raise ValueError(
            f"Pallier2025 runtime context 合同不一致：预期 {expected}，实际 {actual}"
        )
    return dataset_module.RUNTIME_CONTEXT_COLUMN


def frozen_vocabulary(size=SELECTION_VOCABULARY_SIZE):
    manifests, _ = load_vocabulary_assets(VOCABULARY_ASSET_DIRECTORY)
    return tuple(manifests[int(size)]["vocabulary"])


def load_pretrained_brain_encoder(
    model,
    checkpoint_path,
    config,
    channel_names,
    channel_positions,
):
    """只载入同一 Pallier 合同 main_word 检查点中的脑编码器。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "word_decoding/Pallier2025":
        raise ValueError(f"不是 Pallier2025 词解码检查点：{checkpoint_path}")
    if checkpoint.get("dataset") != "pallier2025":
        raise ValueError(f"Pallier2025 检查点数据集身份不一致：{checkpoint_path}")

    source_model = checkpoint.get("model_config", {})
    if bool(source_model.get("use_transformer", True)):
        raise ValueError("Pallier2025 warm-start 必须来自 main_word CNN-only 检查点。")
    if source_model.get("embedding_dimension") != config["model"].get(
        "embedding_dimension"
    ) or source_model.get("conv") != config["model"].get("conv"):
        raise ValueError("Pallier2025 warm-start 的脑编码器结构合同不一致。")
    if checkpoint.get("text_embedding_config") != config["text_embedding"]:
        raise ValueError("Pallier2025 warm-start 的文本表示合同不一致。")

    event_manifest = json.loads(
        (PROJECT_ROOT / "derived/pallier2025/events/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    text_manifest = json.loads(
        (
            PROJECT_ROOT
            / "derived/pallier2025/text/t5_large_layer_0_5/manifest.json"
        ).read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (VOCABULARY_ASSET_DIRECTORY / "run_split.json").read_text(encoding="utf-8")
    )
    source_dataset = checkpoint.get("dataset_contract", {})
    expected_contract = {
        "dataset": "pallier2025",
        "subjects": list(config["dataset"]["subjects"]),
        "subject_order": list(config["dataset"]["subjects"]),
        "runs": list(config["dataset"]["runs"]),
        "split": split_manifest["assignments"],
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "event_table_sha256": event_manifest["event_table_sha256"],
        "window_contract_version": config["dataset"]["window_contract_version"],
        "window_start_offset_seconds": config["dataset"][
            "window_start_offset_seconds"
        ],
        "window_seconds": config["dataset"]["window_seconds"],
        "eligibility_window_seconds": config["dataset"][
            "eligibility_window_seconds"
        ],
        "baseline_seconds": config["dataset"]["baseline_seconds"],
        "clamp": config["dataset"]["clamp"],
        "target_sampling_rate_hz": config["dataset"]["target_sampling_rate_hz"],
        "channel_names_sha256": config["dataset"]["channel_names_sha256"],
        "training_vocabulary_policy": config["dataset"][
            "training_vocabulary_policy"
        ],
        "evaluation_vocabulary_policy": config["dataset"][
            "evaluation_vocabulary_policy"
        ],
        "text_content_sha256": text_manifest["content_sha256"],
    }
    for field, expected in expected_contract.items():
        if source_dataset.get(field) != expected:
            raise ValueError(f"Pallier2025 warm-start 数据合同不一致：{field}")

    if tuple(checkpoint.get("channel_names", ())) != tuple(channel_names):
        raise ValueError("Pallier2025 warm-start 的 MEG 通道顺序不一致。")
    source_positions = np.asarray(checkpoint.get("channel_positions", ()))
    target_positions = np.asarray(channel_positions)
    if source_positions.shape != target_positions.shape or not np.allclose(
        source_positions,
        target_positions,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("Pallier2025 warm-start 的通道位置合同不一致。")

    prefix = "brain_encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError("Pallier2025 warm-start 检查点不包含 brain_encoder 参数。")
    model.brain_encoder.load_state_dict(encoder_state, strict=True)
    source_metrics = checkpoint.get("metrics", {})
    return {
        "method": "main_word_best_checkpoint_brain_encoder_only",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_epoch": int(checkpoint["epoch"]),
        "source_update": int(checkpoint.get("optimizer_updates", 0)),
        "source_macro_r_at_10": source_metrics.get(
            "retrieval_acc10_vocab=pallier2025_50_macro"
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
        frozen_vocabulary(),
        top_ks=top_ks,
        vocabulary_name=SELECTION_VOCABULARY_NAME,
    )
    return metrics, encoded


def train_one_epoch_with_update_budget(
    model,
    loss_module,
    loader,
    optimizer,
    scaler,
    device,
    config,
    maximum_optimizer_updates,
    *,
    scheduler,
):
    """执行固定数量的成功 optimizer updates，并保持 AMP overflow 语义。"""
    model.train()
    loss_module.train()
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    loss_sum = 0.0
    sample_count = 0
    optimizer_updates = 0
    batches_seen = 0
    for batch in loader:
        if optimizer_updates >= int(maximum_optimizer_updates):
            break
        meg, targets, subject_indices, sentence_indices = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            estimates = model(meg, subject_indices, sentence_indices)
            loss = loss_module(estimates, targets)
        scaler.scale(loss).backward()
        max_grad_norm = float(config.get("max_grad_norm", 0.0))
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        previous_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        step_succeeded = not use_amp or float(scaler.get_scale()) >= previous_scale
        if step_succeeded:
            optimizer_updates += 1
            scheduler.step()
        loss_sum += float(loss.detach()) * len(meg)
        sample_count += len(meg)
        batches_seen += 1
    if sample_count == 0:
        raise RuntimeError("未读取任何 Pallier2025 训练 batch。")
    return loss_sum / sample_count, optimizer_updates, batches_seen


def checkpoint_payload(
    model,
    loss_module,
    config,
    epoch,
    metrics,
    channel_names,
    channel_positions,
    *,
    optimizer_updates,
    optimizer=None,
    scheduler=None,
):
    """构造包含完整 Pallier 科学合同的 checkpoint payload。"""
    event_manifest = json.loads(
        (PROJECT_ROOT / "derived/pallier2025/events/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    signal_manifest = json.loads(
        (PROJECT_ROOT / "derived/pallier2025/signals/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    text_manifest = json.loads(
        (
            PROJECT_ROOT
            / "derived/pallier2025/text/t5_large_layer_0_5/manifest.json"
        ).read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (VOCABULARY_ASSET_DIRECTORY / "run_split.json").read_text(encoding="utf-8")
    )
    vocabulary_manifests, _ = load_vocabulary_assets(VOCABULARY_ASSET_DIRECTORY)
    payload = {
        "format_version": 1,
        "task": "word_decoding/Pallier2025",
        "dataset": "pallier2025",
        "epoch": int(epoch),
        "optimizer_updates": int(optimizer_updates),
        "model_state": model.state_dict(),
        "loss_state": loss_module.state_dict(),
        "model_config": config["model"],
        "loss_config": config["loss"],
        "text_embedding_config": config["text_embedding"],
        "metrics": metrics,
        "channel_names": list(channel_names),
        "channel_positions": np.asarray(channel_positions).tolist(),
        "dataset_contract": {
            "dataset": "pallier2025",
            "subjects": list(config["dataset"]["subjects"]),
            "subject_order": list(config["dataset"]["subjects"]),
            "runs": list(config["dataset"]["runs"]),
            "split_manifest": config["dataset"]["split_manifest"],
            "split": split_manifest["assignments"],
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "event_table_sha256": event_manifest["event_table_sha256"],
            "window_contract_version": config["dataset"][
                "window_contract_version"
            ],
            "window_start_offset_seconds": config["dataset"][
                "window_start_offset_seconds"
            ],
            "window_seconds": config["dataset"]["window_seconds"],
            "eligibility_window_seconds": config["dataset"][
                "eligibility_window_seconds"
            ],
            "baseline_seconds": config["dataset"]["baseline_seconds"],
            "clamp": config["dataset"]["clamp"],
            "target_sampling_rate_hz": config["dataset"][
                "target_sampling_rate_hz"
            ],
            "channel_names_sha256": config["dataset"]["channel_names_sha256"],
            "recording_count": signal_manifest["recording_count"],
            "context_grouping": config["dataset"]["context_grouping"],
            "runtime_context_grouping": config["dataset"].get(
                "runtime_context_grouping",
                "not_used_by_word_model",
            ),
            "max_context_words": config["dataset"]["max_context_words"],
            "training_vocabulary_policy": config["dataset"][
                "training_vocabulary_policy"
            ],
            "evaluation_vocabulary_policy": config["dataset"][
                "evaluation_vocabulary_policy"
            ],
            "text_content_sha256": text_manifest["content_sha256"],
            "vocabulary_manifest_sha256": {
                str(size): manifest["manifest_sha256"]
                for size, manifest in vocabulary_manifests.items()
            },
        },
        "provenance": config.get("provenance", {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    return payload


def run_training(config, smoke=False, save=True, force_cache=False):
    """训练 Pallier word/context；只消费 train 与 validation canonical 数据。"""
    del force_cache  # Pallier canonical 训练不允许重建或回退数据缓存。
    training_config = dict(config["training"])
    warm_start = bool(training_config.get("pretrained_brain_encoder_checkpoint"))
    if warm_start and training_config.get("warm_start_from") != "main_word":
        raise ValueError("Pallier2025 context 只能从同 scope 的 main_word warm-start。")
    if warm_start and not bool(config["model"].get("use_transformer", False)):
        raise ValueError("Pallier2025 warm-start 只用于启用 Transformer 的 context 条件。")
    if training_config.get("freeze_brain_encoder_updates") not in (None, 0, ""):
        raise ValueError("Pallier2025 warm-start 不引入额外 brain encoder freeze 阶段。")
    if int(config["model"]["embedding_dimension"]) != int(
        config["text_embedding"]["embedding_dimension"]
    ):
        raise ValueError("Pallier2025 model 与 T5 target 维数不一致。")
    if smoke:
        training_config["epochs"] = 1
        training_config["patience"] = 1
        training_config["batch_size"] = min(
            int(training_config["batch_size"]), 16
        )
        training_config["max_updates"] = 1
        training_config["minimum_updates_before_early_stopping"] = 1

    set_seed(int(training_config["seed"]))
    device = choose_device(training_config.get("device", "auto"))
    event_path = ensure_canonical_inputs(config)
    subjects = config["dataset"]["subjects"]
    train_table = dataset_module.load_event_table(
        event_path,
        split="train",
        subjects=subjects,
        trainable_only=True,
    )
    val_table = dataset_module.load_event_table(
        event_path,
        split="val",
        subjects=subjects,
        trainable_only=True,
    )
    if smoke:
        maximum = int(training_config.get("smoke_rows", 64))
        train_table = limit_rows(train_table, maximum)
        val_table = limit_rows(val_table, maximum)

    train_dataset = build_dataset(config, train_table)
    val_dataset = build_dataset(config, val_table)
    group_column = loader_group_column(config)
    train_loader, train_sampler = make_loader(
        train_dataset,
        training_config,
        shuffle=True,
        group_column=group_column,
    )
    val_loader, _ = make_loader(
        val_dataset,
        training_config,
        shuffle=False,
        group_column=group_column,
    )
    model = build_brain_embedding_model(
        train_dataset.channel_count,
        train_dataset.channel_positions,
        train_dataset.subject_count,
        config["model"],
    ).to(device)
    loss_module = build_siglip_loss(config["loss"]).to(device)
    initialization_audit = None
    if warm_start:
        initialization_audit = load_pretrained_brain_encoder(
            model,
            training_config["pretrained_brain_encoder_checkpoint"],
            config,
            train_dataset.channel_names,
            train_dataset.channel_positions,
        )
    optimizer = build_adamw_for_modules([model, loss_module], training_config)

    max_updates = int(training_config["max_updates"])
    minimum_updates = int(
        training_config.get("minimum_updates_before_early_stopping", 0)
    )
    scheduler_horizon = int(training_config["scheduler_total_updates"])
    if not 0 < minimum_updates <= max_updates <= scheduler_horizon:
        raise ValueError("Pallier2025 update budget / scheduler 合同无效。")
    scheduler = build_cosine_annealing_scheduler(
        optimizer,
        scheduler_horizon,
        minimum_learning_rate=float(
            training_config.get("minimum_learning_rate", 0.0)
        ),
    )
    use_amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    top_ks = tuple(int(value) for value in training_config.get("top_ks", [1, 10]))
    selection_metric = (
        f"retrieval_acc{max(top_ks)}_vocab={SELECTION_VOCABULARY_NAME}_macro"
    )
    if config.get("evaluation", {}).get("selection_metric") != selection_metric:
        raise ValueError("Pallier2025 validation checkpoint selection metric 合同漂移。")
    output_dir = Path(training_config["output_dir"])
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_epoch = None
    best_metrics = None
    best_model_state = None
    best_loss_state = None
    epochs_without_improvement = 0
    optimizer_updates = 0
    stop_reason = "maximum_epochs_reached"
    history = []
    for epoch in range(int(training_config["epochs"])):
        train_sampler.set_epoch(epoch)
        remaining_updates = max_updates - optimizer_updates
        train_loss, epoch_updates, batches_seen = train_one_epoch_with_update_budget(
            model,
            loss_module,
            train_loader,
            optimizer,
            scaler,
            device,
            training_config,
            remaining_updates,
            scheduler=scheduler,
        )
        optimizer_updates += epoch_updates
        validation_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            top_ks=top_ks,
            amp=training_config.get("amp", True),
        )
        score = float(validation_metrics[selection_metric])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_optimizer_updates": int(epoch_updates),
                "optimizer_updates": int(optimizer_updates),
                "batches_seen": int(batches_seen),
                **validation_metrics,
            }
        )
        print(
            f"Epoch {epoch + 1}: loss={train_loss:.5f}, "
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
                        optimizer_updates=optimizer_updates,
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
                    optimizer_updates=optimizer_updates,
                    optimizer=optimizer,
                    scheduler=scheduler,
                ),
            )
        if optimizer_updates >= max_updates:
            stop_reason = "update_budget_reached"
            break
        if (
            epochs_without_improvement >= int(training_config["patience"])
            and optimizer_updates >= minimum_updates
        ):
            stop_reason = "early_stopping"
            break

    if best_epoch is None:
        raise RuntimeError("Pallier2025 训练未产生有效 validation metric。")
    if optimizer_updates < minimum_updates:
        raise RuntimeError(
            f"训练仅完成 {optimizer_updates} 次更新，低于要求的 {minimum_updates} 次。"
        )
    if not save and best_model_state is not None:
        model.load_state_dict(best_model_state)
        loss_module.load_state_dict(best_loss_state)

    summary = {
        "status": "smoke_completed" if smoke else "training_completed",
        "device": str(device),
        "test_meg_opened": False,
        "event_table": str(event_path),
        "train_rows": len(train_dataset),
        "validation_rows": len(val_dataset),
        "meg_shape": [train_dataset.channel_count, train_dataset.window_samples],
        "model_parameter_count": parameter_count(model),
        "loss_parameter_count": parameter_count(loss_module),
        "initialization_audit": initialization_audit,
        "freeze_brain_encoder_updates": 0,
        "text_embedding_contract": config["text_embedding"],
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_validation": best_metrics,
        "optimizer_updates": int(optimizer_updates),
        "target_updates": max_updates,
        "scheduler_total_updates": scheduler_horizon,
        "batch_size_limit": int(training_config["batch_size"]),
        "estimated_batches_per_epoch": len(train_loader),
        "stop_reason": stop_reason,
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
    print("Pallier2025 training completed")
    print(f"  device: {summary['device']}")
    print(f"  train/val: {summary['train_rows']} / {summary['validation_rows']}")
    print(
        f"  best epoch: {summary['best_epoch']}, "
        f"{summary['selection_metric']}: {summary['best_score']:.6f}"
    )
    print(
        f"  optimizer updates: {summary['optimizer_updates']} / "
        f"{summary['target_updates']}"
    )
    print("  test: locked_not_evaluated")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_training(
        load_config(args.config), smoke=args.smoke, save=not args.no_save
    )
    print_training_summary(summary)


if __name__ == "__main__":
    main()
