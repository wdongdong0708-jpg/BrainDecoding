"""以验证集优先的 SMN4Lang 词解码训练入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "SMN4Lang_gpt2.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import SMN4Lang as dataset_module
from losses import build_siglip_loss
from metrics import fixed_vocabulary_retrieval_metrics
from models import build_brain_embedding_model
from optimizers import build_adamw_for_modules, build_cosine_annealing_scheduler
from tasks.word_decoding.LibriBrain100.train import (
    choose_device,
    cpu_state_dict,
    encode_loader,
    limit_rows,
    load_checkpoint,
    make_loader,
    move_batch,
    parameter_count,
    save_checkpoint,
    save_json,
    set_seed,
    train_one_epoch,
)


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(config_path, ancestors=()):
    config_path = Path(config_path).resolve()
    if config_path in ancestors:
        raise ValueError(f"Cyclic config inheritance: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent_config = _load_config_mapping(parent_path, (*ancestors, config_path))
    return _deep_merge(parent_config, config)


def load_config(path=None):
    config_path = project_path(path or DEFAULT_CONFIG)
    config = _load_config_mapping(config_path)
    for key in ("event_table", "meg_dir", "text_embeddings"):
        config["cache"][key] = str(project_path(config["cache"][key]))
    config["training"]["output_dir"] = str(
        project_path(config["training"]["output_dir"])
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
    """构建并审计元数据，然后按需物化明确选择的数据。"""
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
        dataset_module.ensure_configured_text_embedding_cache(
            selected,
            config["dataset"],
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
        "text_embedding_contract": config["text_embedding"],
        "meg_materialized": bool(with_meg_cache),
        "test_meg_opened": bool(with_meg_cache and include_test),
    }


def ensure_event_table(config):
    event_path = Path(config["cache"]["event_table"])
    if not event_path.exists():
        prepare_event_table(config)
    return event_path


def materialize_selected_inputs(config, selected_table, force_cache=False):
    dataset_module.ensure_configured_text_embedding_cache(
        selected_table,
        config["dataset"],
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
    return dataset_module.SMN4LangWordDataset(
        table,
        config["dataset"],
        config["cache"]["meg_dir"],
        config["cache"]["text_embeddings"],
        config["text_embedding"],
        zero_meg=zero_meg,
    )


def train_one_epoch_with_update_budget(
    model,
    loss_module,
    loader,
    optimizer,
    scaler,
    device,
    config,
    maximum_optimizer_updates,
    scheduler=None,
):
    """训练至多执行固定次数的成功优化器更新。"""
    model.train()
    loss_module.train()
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    loss_sum = 0.0
    sample_count = 0
    optimizer_updates = 0
    batches_seen = 0
    maximum_optimizer_updates = int(maximum_optimizer_updates)
    for batch in loader:
        if optimizer_updates >= maximum_optimizer_updates:
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
            if scheduler is not None:
                scheduler.step()
        loss_sum += float(loss.detach()) * len(meg)
        sample_count += len(meg)
        batches_seen += 1
    if sample_count == 0:
        raise RuntimeError("No SMN4Lang training batches were consumed.")
    return loss_sum / sample_count, optimizer_updates, batches_seen


def evaluate_loader(model, loader, device, top_ks=(1, 10), amp=True):
    encoded = encode_loader(model, loader, device, amp=amp)
    metrics = fixed_vocabulary_retrieval_metrics(
        encoded["predictions"],
        encoded["targets"],
        encoded["words"],
        dataset_module.SMN4LANG50_VOCABULARY,
        top_ks=top_ks,
        vocabulary_name="smn4lang50",
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
    optimizer_updates=None,
):
    payload = {
        "format_version": 1,
        "task": "word_decoding/SMN4Lang",
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "loss_state": loss_module.state_dict(),
        "model_config": config["model"],
        "loss_config": config["loss"],
        "text_embedding_config": config["text_embedding"],
        "metrics": metrics,
        "channel_names": list(channel_names),
        "channel_positions": np.asarray(channel_positions).tolist(),
        "dataset_contract": {
            "root": config["dataset"]["root"],
            "snapshot": config["dataset"].get("snapshot"),
            "subjects": config["dataset"]["subjects"],
            "split": config["dataset"]["split"],
            "window_seconds": config["dataset"]["window_seconds"],
            "target_sampling_rate_hz": config["dataset"][
                "target_sampling_rate_hz"
            ],
            "context_grouping": config["dataset"].get("context_grouping"),
            "max_context_words": config["dataset"].get("max_context_words"),
            "training_vocabulary_policy": config["dataset"].get(
                "training_vocabulary_policy"
            ),
            "evaluation_vocabulary_policy": config["dataset"].get(
                "evaluation_vocabulary_policy"
            ),
            "evaluation_vocabulary": list(dataset_module.SMN4LANG50_VOCABULARY),
            "vocabulary": list(dataset_module.SMN4LANG50_VOCABULARY),
            "timing": {
                "fmri_offset_removed_seconds": (
                    dataset_module.FMRI_ALIGNMENT_OFFSET_SECONDS
                ),
                "meg_audio_delay_added_seconds": (
                    dataset_module.MEG_AUDIO_DELAY_SECONDS
                ),
            },
        },
        "provenance": config.get("provenance", {}),
    }
    if optimizer_updates is not None:
        payload["optimizer_updates"] = int(optimizer_updates)
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    return payload


def run_training(config, smoke=False, save=True, force_cache=False):
    training_config = dict(config["training"])
    model_dimension = int(config["model"]["embedding_dimension"])
    target_dimension = int(config["text_embedding"]["embedding_dimension"])
    if model_dimension != target_dimension:
        raise ValueError(
            "Model and text-target dimensions differ: "
            f"{model_dimension} != {target_dimension}."
        )
    if smoke:
        training_config["epochs"] = 1
        training_config["patience"] = 1
        training_config["batch_size"] = min(
            int(training_config["batch_size"]), 16
        )
        if training_config.get("max_updates") is not None:
            training_config["max_updates"] = 1
            training_config["minimum_updates_before_early_stopping"] = 1
    set_seed(int(training_config["seed"]))
    device = choose_device(training_config.get("device", "auto"))
    event_path = ensure_event_table(config)
    train_table = dataset_module.load_event_table(event_path, split="train")
    val_table = dataset_module.load_event_table(event_path, split="val")
    if smoke:
        maximum = int(training_config.get("smoke_rows", 64))
        train_table = limit_rows(train_table, maximum)
        val_table = limit_rows(val_table, maximum)

    materialize_selected_inputs(
        config,
        pd.concat([train_table, val_table], ignore_index=True),
        force_cache=force_cache,
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
    optimizer = build_adamw_for_modules([model, loss_module], training_config)
    max_updates = training_config.get("max_updates")
    if max_updates is not None:
        max_updates = int(max_updates)
        if max_updates <= 0:
            raise ValueError("training.max_updates must be positive.")
    minimum_updates = int(
        training_config.get("minimum_updates_before_early_stopping", 0)
    )
    if minimum_updates < 0:
        raise ValueError("minimum_updates_before_early_stopping cannot be negative.")
    if max_updates is not None and minimum_updates > max_updates:
        raise ValueError(
            "minimum_updates_before_early_stopping cannot exceed max_updates."
        )
    scheduler_horizon = int(
        training_config.get("scheduler_total_updates", max_updates)
        if max_updates is not None
        else training_config["epochs"]
    )
    if scheduler_horizon <= 0:
        raise ValueError("The cosine scheduler horizon must be positive.")
    if max_updates is not None and scheduler_horizon < max_updates:
        raise ValueError("scheduler_total_updates cannot be smaller than max_updates.")
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
    selection_metric = f"retrieval_acc{max(top_ks)}_vocab=smn4lang50_macro"
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
        if max_updates is None:
            train_loss = train_one_epoch(
                model,
                loss_module,
                train_loader,
                optimizer,
                scaler,
                device,
                training_config,
            )
            epoch_optimizer_updates = len(train_loader)
            batches_seen = len(train_loader)
        else:
            remaining_updates = max_updates - optimizer_updates
            train_loss, epoch_optimizer_updates, batches_seen = (
                train_one_epoch_with_update_budget(
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
            )
        optimizer_updates += epoch_optimizer_updates
        validation_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            top_ks=top_ks,
            amp=training_config.get("amp", True),
        )
        if max_updates is None:
            scheduler.step()
        score = float(validation_metrics[selection_metric])
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_optimizer_updates": int(epoch_optimizer_updates),
            "optimizer_updates": int(optimizer_updates),
            "batches_seen": int(batches_seen),
            **validation_metrics,
        }
        history.append(epoch_record)
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
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_updates=optimizer_updates,
                ),
            )
        if max_updates is not None and optimizer_updates >= max_updates:
            stop_reason = "update_budget_reached"
            break
        if (
            epochs_without_improvement >= int(training_config["patience"])
            and optimizer_updates >= minimum_updates
        ):
            stop_reason = "early_stopping"
            break

    if best_epoch is None:
        raise RuntimeError("Training did not produce a valid validation metric.")
    if optimizer_updates < minimum_updates:
        raise RuntimeError(
            f"Training stopped after {optimizer_updates} updates before the required "
            f"minimum of {minimum_updates}. Increase training.epochs."
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
        save_json(output_dir / "training_summary.json", summary)
    return summary


def print_training_summary(summary):
    print("SMN4Lang training completed")
    print(f"  device: {summary['device']}")
    print(f"  train/val: {summary['train_rows']} / {summary['validation_rows']}")
    print(
        f"  best epoch: {summary['best_epoch']}, "
        f"{summary['selection_metric']}: {summary['best_score']:.6f}"
    )
    print(
        f"  optimizer updates: {summary['optimizer_updates']} / "
        f"{summary['target_updates'] or 'epoch-scheduled'}"
    )
    print("  test: locked_not_evaluated")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--with-meg-cache", action="store_true")
    parser.add_argument("--with-text-embeddings", action="store_true")
    parser.add_argument("--include-test", action="store_true")
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
    summary = run_training(
        config,
        smoke=args.smoke,
        save=not args.no_save,
        force_cache=args.force_cache,
    )
    print_training_summary(summary)


if __name__ == "__main__":
    main()
