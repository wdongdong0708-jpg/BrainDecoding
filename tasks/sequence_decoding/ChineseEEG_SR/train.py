"""ChineseEEG 行级文本检索训练入口。"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

TASK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_DIR.parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ChineseEEG_SR.yaml"

# 支持直接运行当前任务的 train.py。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import ChineseEEG_SR as dataset_module
from losses import multi_positive_contrastive_loss
from metrics import (
    paired_cluster_bootstrap,
    retrieval_metrics,
    retrieval_metrics_with_ranks,
    summarize_retrieval,
)
from models import build_strided_conv_encoder
from optimizers import build_optimizer, build_scheduler


def project_path(value):
    """把配置中的相对路径统一锚定到项目根目录。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path=None):
    """读取任务配置并解析项目内路径。"""
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["cache_dir"] = str(project_path(config.get("cache_dir", "cache")))
    for section, key in (
        ("training", "event_table"),
        ("training", "output_dir"),
        ("closed_set_diagnostic", "output_dir"),
    ):
        if section in config and key in config[section]:
            config[section][key] = str(project_path(config[section][key]))
    return config


def load_dataset_module(dataset_name):
    """返回当前任务绑定的 ChineseEEG 静默阅读数据模块。"""
    if dataset_name not in {"ChineseEEG1_SR", "ChineseEEG_SR"}:
        raise ValueError(f"当前训练入口不支持数据集：{dataset_name}")
    return dataset_module


def save_event_table(event_table, novel, cache_dir):
    """将带划分字段的事件表保存为 Excel。"""
    output_path = Path(cache_dir) / f"ChineseEEG_{novel}_run_split.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_table.to_excel(output_path, index=False, sheet_name="行级事件表")
    return output_path


def prepare_event_table(config):
    """构造统一行级事件表并按 run 划分。"""
    event_table = dataset_module.build_event_table(config["dataset"])
    event_table = dataset_module.split_event_table(event_table, config["split"])
    dataset_module.audit_run_split(event_table)
    event_table = dataset_module.rename_fields(event_table, config.get("field_names"))
    return save_event_table(
        event_table,
        novel=config["dataset"].get("novel", "all"),
        cache_dir=config["cache_dir"],
    )


def set_seed(seed):
    """固定主要随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    """选择训练设备；auto 优先使用 CUDA。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前环境不可用。")
    return device


def limit_rows(table, maximum):
    """冒烟测试只保留少量记录，正式训练不截断。"""
    if maximum is None or len(table) <= maximum:
        return table.reset_index(drop=True)
    return table.iloc[: int(maximum)].reset_index(drop=True)


def channel_moments(dataset):
    """累计一个数据集的逐通道一阶、二阶矩。"""
    channel_sum = np.zeros(len(dataset.channel_names), dtype=np.float64)
    channel_square_sum = np.zeros_like(channel_sum)
    sample_count = 0
    for index in range(len(dataset)):
        eeg = dataset.read_eeg(index, normalize=False).astype(np.float64, copy=False)
        channel_sum += eeg.sum(axis=1)
        channel_square_sum += np.square(eeg).sum(axis=1)
        sample_count += eeg.shape[1]
        if (index + 1) % 500 == 0:
            print(f"标准化统计：{index + 1}/{len(dataset)}")
    return channel_sum, channel_square_sum, sample_count


def normalization_from_moments(moments):
    """合并训练被试矩并生成标准化参数。"""
    channel_sum = sum(moment[0] for moment in moments)
    channel_square_sum = sum(moment[1] for moment in moments)
    sample_count = sum(moment[2] for moment in moments)

    mean = channel_sum / sample_count
    variance = channel_square_sum / sample_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-8))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError("训练集标准化参数含非有限值。")
    return mean.astype(np.float32), std.astype(np.float32)


def estimate_channel_stats(dataset):
    """仅用训练 EEG 估计逐通道均值和标准差。"""
    return normalization_from_moments([channel_moments(dataset)])


def make_loader(dataset, config, shuffle):
    """构造 DataLoader；Windows 默认使用单进程读取 BrainVision。"""
    generator = torch.Generator().manual_seed(int(config["seed"]))
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def move_batch(batch, device):
    """只把训练所需张量移动到计算设备。"""
    return (
        batch["eeg"].to(device, non_blocking=True),
        batch["text_embedding"].to(device, non_blocking=True),
        batch["text_key"].to(device, non_blocking=True),
        batch["subject_index"].to(device, non_blocking=True),
    )


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, config):
    """训练一个 epoch。"""
    model.train()
    loss_sum = 0.0
    sample_count = 0
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    for batch in loader:
        eeg, text_embedding, text_keys, subject_indices = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            eeg_embedding = model(eeg, subject_indices)
            loss = multi_positive_contrastive_loss(
                eeg_embedding,
                text_embedding,
                text_keys,
                temperature=float(config["temperature"]),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # AMP 溢出时优化器会跳步，此时学习率也不应前进。
        if not use_amp or scaler.get_scale() >= scale_before_step:
            scheduler.step()
        batch_size = len(eeg)
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size
    return loss_sum / sample_count


@torch.inference_mode()
def encode_loader(model, loader, device):
    """编码整个验证集，并保留其配对 BERT 目标。"""
    model.eval()
    eeg_embeddings = []
    text_embeddings = []
    text_ids = []
    metadata = {
        "subject_id": [],
        "recording_id": [],
        "run_id": [],
        "row_id": [],
    }
    for batch in loader:
        eeg = batch["eeg"].to(device, non_blocking=True)
        subject_indices = batch["subject_index"].to(device, non_blocking=True)
        eeg_embeddings.append(model(eeg, subject_indices).float().cpu().numpy())
        text_embeddings.append(batch["text_embedding"].float().numpy())
        text_ids.extend(str(value) for value in batch["text_id"])
        metadata["subject_id"].extend(str(value) for value in batch["subject_id"])
        metadata["recording_id"].extend(str(value) for value in batch["recording_id"])
        metadata["run_id"].extend(int(value) for value in batch["run_id"])
        metadata["row_id"].extend(int(value) for value in batch["row_id"])
    return (
        np.concatenate(eeg_embeddings),
        np.concatenate(text_embeddings),
        np.asarray(text_ids),
        metadata,
    )


def evaluate_model(model, loader, device, top_ks):
    """在当前验证候选池上计算检索指标。"""
    eeg_embeddings, text_embeddings, text_ids, _ = encode_loader(model, loader, device)
    return retrieval_metrics(eeg_embeddings, text_embeddings, text_ids, top_ks=top_ks)


def evaluate_model_with_ranks(model, loader, device, top_ks):
    """返回验证指标、逐查询排名和查询标识。"""
    eeg_embeddings, text_embeddings, text_ids, metadata = encode_loader(
        model, loader, device
    )
    metrics, ranks = retrieval_metrics_with_ranks(
        eeg_embeddings, text_embeddings, text_ids, top_ks=top_ks
    )
    metadata["text_id"] = text_ids.tolist()
    return metrics, ranks, metadata


def cpu_state_dict(model):
    """复制一份可移植的模型参数。"""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    mean,
    std,
    channel_names,
    channel_positions,
    subject_to_index,
    model_config,
    metrics,
):
    """保存可恢复的训练状态。"""
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "channel_mean": mean,
            "channel_std": std,
            "channel_names": list(channel_names),
            "channel_positions": np.asarray(channel_positions, dtype=np.float32),
            "subject_to_index": dict(subject_to_index),
            "model_config": dict(model_config),
            "validation_metrics": metrics,
        },
        path,
    )


def json_value(value):
    """把 NumPy 标量转换为 JSON 原生类型。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"不支持写入 JSON 的类型：{type(value).__name__}")


def save_summary(path, summary):
    """保存便于审计的训练摘要。"""
    content = json.dumps(summary, ensure_ascii=False, indent=2, default=json_value)
    path.write_text(content, encoding="utf-8")


def format_percentage(value):
    """把 0 到 1 的指标显示为百分比。"""
    return f"{float(value):.2%}"


def print_retrieval_metrics(title, metrics, top_ks):
    """用中文打印一组检索指标。"""
    print(f"\n{title}")
    print(f"  查询数：{metrics['query_count']}")
    print(f"  候选文本数：{metrics['candidate_count']}")
    print(f"  目标文本数：{metrics['target_count']}")
    print(f"  中位排名：{metrics['median_rank']:.2f}")
    print(f"  平均倒数排名：{format_percentage(metrics['mean_reciprocal_rank'])}")
    for k in top_ks:
        print(
            f"  R@{k}：宏平均 {format_percentage(metrics[f'macro_recall_at_{k}'])}，"
            f"微平均 {format_percentage(metrics[f'micro_recall_at_{k}'])}，"
            f"随机水平 {format_percentage(metrics[f'random_recall_at_{k}'])}"
        )


def print_training_summary(summary, top_ks):
    """用中文打印最终训练摘要。"""
    status_names = {"smoke_completed": "冒烟测试完成", "completed": "训练完成"}
    print("\n训练摘要")
    print(f"  状态：{status_names.get(summary['status'], summary['status'])}")
    print(f"  设备：{summary['device']}")
    print(f"  训练样本数：{summary['train_rows']}")
    print(f"  验证样本数：{summary['validation_rows']}")
    print(f"  EEG形状：{summary['eeg_shape'][0]} × {summary['eeg_shape'][1]}")
    print(f"  受试者数：{summary['subject_count']}")
    print(f"  傅里叶虚拟通道数：{summary['model'].get('virtual_channels', 64)}")
    print(f"  傅里叶阶数：{summary['model'].get('fourier_harmonics', 8)}")
    print(f"  测试EEG已打开：{'是' if summary['test_eeg_opened'] else '否'}")
    print(
        f"  最佳验证宏平均 R@{max(top_ks)}："
        f"{format_percentage(summary['best_score'])}"
    )
    validation = summary["validation"]
    print(f"  候选集定义：{validation['candidate_definition']}")
    print_retrieval_metrics("正确配对验证", validation["clean"], top_ks)
    print_retrieval_metrics("同记录错行EEG对照", validation["same_recording_wrong_row"], top_ks)
    print_retrieval_metrics("零EEG对照", validation["zero_eeg"], top_ks)


def print_paired_analysis(analysis, top_ks):
    """用中文打印配对 bootstrap 结果。"""
    print("\n验证集配对分析")
    print(f"  检查点：{analysis['checkpoint']}")
    print(f"  查询数：{analysis['query_count']}")
    print(f"  物理显示行数：{analysis['physical_row_count']}")
    print(f"  测试EEG已打开：{'是' if analysis['test_eeg_opened'] else '否'}")
    print(f"  测试解锁门槛：{'通过' if analysis['test_unlock_gate'] else '未通过'}")
    print(f"  决策：{analysis['decision']}")
    names = {
        "same_recording_wrong_row": "正确配对减同记录错行EEG",
        "zero_eeg": "正确配对减零EEG",
    }
    for control_name, result in analysis["paired_bootstrap"].items():
        print(f"\n{names[control_name]}")
        for k in top_ks:
            values = result["recall_differences"][f"macro_recall_at_{k}"]
            print(
                f"  宏平均 R@{k} 差值：{format_percentage(values['estimate'])}，"
                f"95% CI [{format_percentage(values['ci_95_low'])}, "
                f"{format_percentage(values['ci_95_high'])}]"
            )


def build_subject_index(train_table, compared_table=None):
    """仅由训练表建立稳定的受试者索引。"""
    subjects = sorted(str(value) for value in train_table["被试编号"].unique())
    mapping = {subject: index for index, subject in enumerate(subjects)}
    if compared_table is not None:
        compared_subjects = {str(value) for value in compared_table["被试编号"].unique()}
        unknown = sorted(compared_subjects - set(mapping))
        if unknown:
            raise ValueError(f"验证集含训练中未出现的被试：{unknown}")
    return mapping


def build_datasets(config, smoke=False):
    """只构造训练集和验证集，不实例化测试集。"""
    dataset_module = load_dataset_module(config["dataset"]["name"])
    event_table_path = Path(config["training"]["event_table"])
    train_table = dataset_module.load_split_event_table(event_table_path, "train")
    val_table = dataset_module.load_split_event_table(event_table_path, "val")
    subject_to_index = build_subject_index(train_table, val_table)
    if smoke:
        maximum = int(config["training"].get("smoke_rows", 32))
        train_table = limit_rows(train_table, maximum)
        val_table = limit_rows(val_table, maximum)

    train_dataset = dataset_module.ChineseEEGRowDataset(
        train_table, config["dataset"], subject_to_index=subject_to_index
    )
    val_dataset = dataset_module.ChineseEEGRowDataset(
        val_table, config["dataset"], subject_to_index=subject_to_index
    )
    if train_dataset.channel_names != val_dataset.channel_names:
        raise ValueError("训练集与验证集通道顺序不一致。")
    if not np.allclose(train_dataset.channel_positions, val_dataset.channel_positions):
        raise ValueError("训练集与验证集电极坐标不一致。")
    return dataset_module, train_dataset, val_dataset


def build_control_datasets(dataset_module, val_dataset, mean, std):
    """构造同 recording 错行和零 EEG 验证对照。"""
    wrong_indices = dataset_module.build_wrong_row_indices(val_dataset.table)
    wrong_dataset = dataset_module.ChineseEEGRowDataset(
        val_dataset.table,
        val_dataset.config,
        source_indices=wrong_indices,
        subject_to_index=val_dataset.subject_to_index,
    )
    zero_dataset = dataset_module.ChineseEEGRowDataset(
        val_dataset.table,
        val_dataset.config,
        zero_eeg=True,
        subject_to_index=val_dataset.subject_to_index,
    )
    wrong_dataset.set_normalization(mean, std)
    zero_dataset.set_normalization(mean, std)
    return wrong_dataset, zero_dataset


def assert_same_queries(reference, compared):
    """确认不同对照严格使用同一批验证查询。"""
    for field in ("subject_id", "recording_id", "run_id", "row_id", "text_id"):
        if reference[field] != compared[field]:
            raise RuntimeError(f"验证对照的 {field} 顺序不一致。")


def build_query_records(metadata, clean_ranks, wrong_ranks, zero_ranks):
    """整理逐查询排名，便于后续复核。"""
    records = []
    for index in range(len(clean_ranks)):
        records.append(
            {
                "被试编号": metadata["subject_id"][index],
                "记录编号": metadata["recording_id"][index],
                "Run编号": metadata["run_id"][index],
                "行编号": metadata["row_id"][index],
                "文本ID": metadata["text_id"][index],
                "正确配对排名": float(clean_ranks[index]),
                "同记录错行EEG排名": float(wrong_ranks[index]),
                "零EEG排名": float(zero_ranks[index]),
            }
        )
    return records


def evaluate_checkpoint(config, checkpoint_path, repetitions=2000, save=True, smoke=False):
    """复用已有检查点完成验证集逐查询配对分析。"""
    training_config = config["training"]
    top_ks = tuple(int(k) for k in training_config.get("top_ks", [1, 5, 10]))
    device = choose_device(training_config.get("device", "auto"))
    dataset_module = load_dataset_module(config["dataset"]["name"])
    val_table = dataset_module.load_split_event_table(
        training_config["event_table"], "val"
    )
    if smoke:
        val_table = limit_rows(val_table, int(training_config.get("smoke_rows", 32)))
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"subject_to_index", "channel_positions", "model_config"}
    if not required.issubset(checkpoint):
        raise ValueError("检查点不含傅里叶空间层所需的受试者索引或电极坐标。")
    val_dataset = dataset_module.ChineseEEGRowDataset(
        val_table,
        config["dataset"],
        subject_to_index=checkpoint["subject_to_index"],
    )
    if tuple(checkpoint["channel_names"]) != val_dataset.channel_names:
        raise ValueError("检查点通道顺序与验证集不一致。")
    if not np.allclose(checkpoint["channel_positions"], val_dataset.channel_positions):
        raise ValueError("检查点电极坐标与验证集不一致。")
    mean = np.asarray(checkpoint["channel_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["channel_std"], dtype=np.float32)
    val_dataset.set_normalization(mean, std)

    model_config = dict(checkpoint["model_config"])
    model_config.setdefault("embedding_dimension", config["dataset"]["embedding_dimension"])
    model = build_strided_conv_encoder(
        len(val_dataset.channel_names),
        val_dataset.channel_positions,
        val_dataset.subject_count,
        model_config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    wrong_dataset, zero_dataset = build_control_datasets(
        dataset_module, val_dataset, mean, std
    )
    clean = evaluate_model_with_ranks(
        model, make_loader(val_dataset, training_config, shuffle=False), device, top_ks
    )
    wrong = evaluate_model_with_ranks(
        model, make_loader(wrong_dataset, training_config, shuffle=False), device, top_ks
    )
    zero = evaluate_model_with_ranks(
        model, make_loader(zero_dataset, training_config, shuffle=False), device, top_ks
    )
    clean_metrics, clean_ranks, metadata = clean
    wrong_metrics, wrong_ranks, wrong_metadata = wrong
    zero_metrics, zero_ranks, zero_metadata = zero
    assert_same_queries(metadata, wrong_metadata)
    assert_same_queries(metadata, zero_metadata)

    cluster_ids = np.asarray(
        [
            f"run-{run_id:02d}_row-{row_id:04d}"
            for run_id, row_id in zip(metadata["run_id"], metadata["row_id"])
        ]
    )
    text_ids = np.asarray(metadata["text_id"])
    seed = int(training_config["seed"])
    paired_bootstrap = {
        "same_recording_wrong_row": paired_cluster_bootstrap(
            clean_ranks,
            wrong_ranks,
            text_ids,
            cluster_ids,
            top_ks=top_ks,
            repetitions=repetitions,
            seed=seed,
        ),
        "zero_eeg": paired_cluster_bootstrap(
            clean_ranks,
            zero_ranks,
            text_ids,
            cluster_ids,
            top_ks=top_ks,
            repetitions=repetitions,
            seed=seed,
        ),
    }
    primary_metric = f"macro_recall_at_{max(top_ks)}"
    primary_interval = paired_bootstrap["same_recording_wrong_row"][
        "recall_differences"
    ][primary_metric]
    test_unlock_gate = primary_interval["ci_95_low"] > 0
    analysis = {
        "status": "validation_only_selected_checkpoint_diagnostic",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_eeg_opened": False,
        "query_count": int(len(clean_ranks)),
        "physical_row_count": int(len(np.unique(cluster_ids))),
        "selection_note": "检查点由同一验证集的宏平均R@10选择，差值可能偏乐观。",
        "primary_metric": primary_metric,
        "test_unlock_gate": test_unlock_gate,
        "decision": "可以解锁测试集" if test_unlock_gate else "保持测试集锁定",
        "metrics": {
            "clean": clean_metrics,
            "same_recording_wrong_row": wrong_metrics,
            "zero_eeg": zero_metrics,
        },
        "paired_bootstrap": paired_bootstrap,
        "queries": build_query_records(
            metadata, clean_ranks, wrong_ranks, zero_ranks
        ),
    }
    if save:
        save_summary(checkpoint_path.parent / "validation_paired_analysis.json", analysis)
    return analysis, top_ks


def closed_set_support(dataset_module, config, smoke=False):
    """选择训练 run 中记录完整且共享同一批文本的被试。"""
    table = dataset_module.load_split_event_table(
        config["training"]["event_table"], "train"
    )
    all_runs = set(table["Run编号"].unique())
    subject_runs = table.groupby("被试编号")["Run编号"].agg(lambda values: set(values))
    subjects = sorted(subject_runs[subject_runs.map(lambda runs: runs == all_runs)].index)
    if len(subjects) < 4:
        raise ValueError("完整被试不足 4 名，不能进行嵌套留出被试诊断。")

    common_ids = None
    for subject in subjects:
        identifiers = set(table.loc[table["被试编号"].eq(subject), "文本ID"])
        common_ids = identifiers if common_ids is None else common_ids & identifiers
    reference = table[table["被试编号"].eq(subjects[0])]
    ordered_ids = [identifier for identifier in reference["文本ID"] if identifier in common_ids]
    if smoke:
        maximum = int(config["closed_set_diagnostic"].get("smoke_texts", 32))
        ordered_ids = ordered_ids[:maximum]
    selected = table[
        table["被试编号"].isin(subjects) & table["文本ID"].isin(ordered_ids)
    ].copy()
    selected["文本顺序"] = selected["文本ID"].map(
        {identifier: index for index, identifier in enumerate(ordered_ids)}
    )
    selected = selected.sort_values(["被试编号", "文本顺序"]).drop(columns="文本顺序")
    if not selected.groupby("被试编号").size().eq(len(ordered_ids)).all():
        raise RuntimeError("闭集共同文本支持不一致。")
    source_runs = [int(run) for run in sorted(all_runs)]
    return selected.reset_index(drop=True), subjects, ordered_ids, source_runs


def fit_closed_set_fold(train_dataset, selection_dataset, training_config, model_config, seed):
    """在内层被试上选 epoch，并返回冻结的最佳模型。"""
    set_seed(seed)
    fold_config = dict(training_config)
    fold_config["seed"] = seed
    train_loader = make_loader(train_dataset, fold_config, shuffle=True)
    selection_loader = make_loader(selection_dataset, fold_config, shuffle=False)
    model = build_strided_conv_encoder(
        len(train_dataset.channel_names),
        train_dataset.channel_positions,
        train_dataset.subject_count,
        model_config,
    ).to(choose_device(fold_config.get("device", "auto")))
    device = next(model.parameters()).device
    optimizer = build_optimizer(model, fold_config)
    total_steps = max(len(train_loader) * int(fold_config["epochs"]), 1)
    scheduler = build_scheduler(
        optimizer,
        total_steps,
        warmup_ratio=float(fold_config.get("warmup_ratio", 0.05)),
    )
    use_amp = bool(fold_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    top_ks = tuple(int(k) for k in fold_config["top_ks"])
    selection_metric = f"macro_recall_at_{max(top_ks)}"
    best_score = -float("inf")
    best_state = None
    best_epoch = None
    stale_epochs = 0
    history = []

    for epoch in range(int(fold_config["epochs"])):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, fold_config
        )
        metrics = evaluate_model(model, selection_loader, device, top_ks)
        score = metrics[selection_metric]
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})
        if epoch > 0:
            print()
        print(f"  内层第 {epoch + 1} 轮")
        print(f"    训练损失：{train_loss:.4f}")
        print(
            f"    内层宏平均 R@{max(top_ks)}："
            f"{format_percentage(score)}"
        )
        if score > best_score:
            best_score = score
            best_state = cpu_state_dict(model)
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(fold_config["patience"]):
            break

    model.load_state_dict(best_state)
    return model, device, best_epoch, best_score, history


def print_closed_set_summary(summary, top_ks):
    """打印闭集嵌套留出被试诊断结果。"""
    print("\n闭集留出被试诊断摘要")
    print(f"  完整被试数：{summary['subject_count']}")
    print(f"  共同文本数：{summary['candidate_count']}")
    print(f"  外层查询数：{summary['query_count']}")
    print(f"  原验证run EEG已打开：{'是' if summary['validation_eeg_opened'] else '否'}")
    print(f"  测试EEG已打开：{'是' if summary['test_eeg_opened'] else '否'}")
    print(f"  诊断门槛：{'通过' if summary['diagnostic_gate'] else '未通过'}")
    print(f"  结论：{summary['decision']}")
    print_retrieval_metrics("外层正确配对", summary["metrics"]["clean"], top_ks)
    print_retrieval_metrics(
        "外层同记录错行EEG", summary["metrics"]["same_recording_wrong_row"], top_ks
    )
    print_retrieval_metrics("外层零EEG", summary["metrics"]["zero_eeg"], top_ks)
    names = {
        "same_recording_wrong_row": "正确配对减同记录错行EEG",
        "zero_eeg": "正确配对减零EEG",
    }
    for control_name, result in summary["paired_bootstrap"].items():
        print(f"\n{names[control_name]}")
        for k in top_ks:
            values = result["recall_differences"][f"macro_recall_at_{k}"]
            print(
                f"  宏平均 R@{k} 差值：{format_percentage(values['estimate'])}，"
                f"95% CI [{format_percentage(values['ci_95_low'])}, "
                f"{format_percentage(values['ci_95_high'])}]"
            )


def run_closed_set_loso(config, smoke=False, save=True):
    """运行训练 run 内的嵌套留出被试闭集诊断。"""
    dataset_module = load_dataset_module(config["dataset"]["name"])
    table, subjects, text_ids, source_runs = closed_set_support(
        dataset_module, config, smoke=smoke
    )
    subject_to_index = {subject: index for index, subject in enumerate(subjects)}
    training_config = dict(config["training"])
    if smoke:
        training_config["epochs"] = 1
        training_config["patience"] = 1
        training_config["batch_size"] = min(int(training_config["batch_size"]), 8)
    top_ks = tuple(int(k) for k in training_config["top_ks"])
    model_config = dict(config["model"])

    print("计算各被试通道统计")
    subject_moments = {}
    channel_names = None
    for subject in subjects:
        print(f"  {subject}")
        subject_table = table[table["被试编号"].eq(subject)].reset_index(drop=True)
        subject_dataset = dataset_module.ChineseEEGRowDataset(
            subject_table,
            config["dataset"],
            subject_to_index=subject_to_index,
        )
        if channel_names is None:
            channel_names = subject_dataset.channel_names
        elif subject_dataset.channel_names != channel_names:
            raise ValueError("闭集被试的 EEG 通道顺序不一致。")
        subject_moments[subject] = channel_moments(subject_dataset)

    fold_summaries = []
    all_records = []
    all_clean_ranks = []
    all_wrong_ranks = []
    all_zero_ranks = []
    all_text_ids = []
    base_seed = int(training_config["seed"])

    for fold_index, outer_subject in enumerate(subjects):
        inner_subject = subjects[(fold_index + 1) % len(subjects)]
        train_subjects = [
            subject
            for subject in subjects
            if subject not in {outer_subject, inner_subject}
        ]
        print(f"\n外层折 {fold_index + 1}/{len(subjects)}")
        print(f"  外层留出被试：{outer_subject}")
        print(f"  内层选模被试：{inner_subject}")
        mean, std = normalization_from_moments(
            [subject_moments[subject] for subject in train_subjects]
        )
        train_table = table[table["被试编号"].isin(train_subjects)].reset_index(drop=True)
        inner_table = table[table["被试编号"].eq(inner_subject)].reset_index(drop=True)
        outer_table = table[table["被试编号"].eq(outer_subject)].reset_index(drop=True)
        train_dataset = dataset_module.ChineseEEGRowDataset(
            train_table,
            config["dataset"],
            subject_to_index=subject_to_index,
        )
        inner_dataset = dataset_module.ChineseEEGRowDataset(
            inner_table,
            config["dataset"],
            subject_to_index=subject_to_index,
        )
        outer_dataset = dataset_module.ChineseEEGRowDataset(
            outer_table,
            config["dataset"],
            subject_to_index=subject_to_index,
        )
        for dataset in (train_dataset, inner_dataset, outer_dataset):
            dataset.set_normalization(mean, std)

        fold_seed = base_seed + fold_index
        model, device, best_epoch, best_score, history = fit_closed_set_fold(
            train_dataset,
            inner_dataset,
            training_config,
            model_config,
            fold_seed,
        )
        wrong_dataset, zero_dataset = build_control_datasets(
            dataset_module, outer_dataset, mean, std
        )
        clean = evaluate_model_with_ranks(
            model, make_loader(outer_dataset, training_config, False), device, top_ks
        )
        wrong = evaluate_model_with_ranks(
            model, make_loader(wrong_dataset, training_config, False), device, top_ks
        )
        zero = evaluate_model_with_ranks(
            model, make_loader(zero_dataset, training_config, False), device, top_ks
        )
        clean_metrics, clean_ranks, metadata = clean
        wrong_metrics, wrong_ranks, wrong_metadata = wrong
        zero_metrics, zero_ranks, zero_metadata = zero
        assert_same_queries(metadata, wrong_metadata)
        assert_same_queries(metadata, zero_metadata)
        print(
            f"  外层宏平均 R@{max(top_ks)}："
            f"{format_percentage(clean_metrics[f'macro_recall_at_{max(top_ks)}'])}"
        )
        fold_summaries.append(
            {
                "fold": fold_index + 1,
                "outer_subject": outer_subject,
                "inner_subject": inner_subject,
                "train_subjects": train_subjects,
                "seed": fold_seed,
                "best_epoch": best_epoch,
                "best_inner_score": best_score,
                "history": history,
                "metrics": {
                    "clean": clean_metrics,
                    "same_recording_wrong_row": wrong_metrics,
                    "zero_eeg": zero_metrics,
                },
            }
        )
        all_records.extend(
            build_query_records(metadata, clean_ranks, wrong_ranks, zero_ranks)
        )
        all_clean_ranks.extend(clean_ranks.tolist())
        all_wrong_ranks.extend(wrong_ranks.tolist())
        all_zero_ranks.extend(zero_ranks.tolist())
        all_text_ids.extend(metadata["text_id"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    clean_ranks = np.asarray(all_clean_ranks)
    wrong_ranks = np.asarray(all_wrong_ranks)
    zero_ranks = np.asarray(all_zero_ranks)
    repeated_text_ids = np.asarray(all_text_ids)
    candidate_count = len(text_ids)
    aggregate_metrics = {
        "clean": summarize_retrieval(
            clean_ranks, repeated_text_ids, candidate_count, top_ks
        ),
        "same_recording_wrong_row": summarize_retrieval(
            wrong_ranks, repeated_text_ids, candidate_count, top_ks
        ),
        "zero_eeg": summarize_retrieval(
            zero_ranks, repeated_text_ids, candidate_count, top_ks
        ),
    }
    repetitions = int(config["closed_set_diagnostic"]["bootstrap_repetitions"])
    paired = {
        "same_recording_wrong_row": paired_cluster_bootstrap(
            clean_ranks,
            wrong_ranks,
            repeated_text_ids,
            repeated_text_ids,
            top_ks,
            repetitions,
            base_seed,
        ),
        "zero_eeg": paired_cluster_bootstrap(
            clean_ranks,
            zero_ranks,
            repeated_text_ids,
            repeated_text_ids,
            top_ks,
            repetitions,
            base_seed,
        ),
    }
    primary_metric = f"macro_recall_at_{max(top_ks)}"
    diagnostic_gate = all(
        result["recall_differences"][primary_metric]["ci_95_low"] > 0
        for result in paired.values()
    )
    summary = {
        "status": "smoke_completed" if smoke else "completed",
        "protocol": "train_only_nested_leave_one_subject_out_closed_set",
        "source_runs": source_runs,
        "subjects": subjects,
        "subject_count": len(subjects),
        "candidate_count": candidate_count,
        "query_count": len(all_clean_ranks),
        "validation_eeg_opened": False,
        "test_eeg_opened": False,
        "selection_rule": "每折由固定的内层留出被试选择宏平均R@10最佳epoch",
        "primary_metric": primary_metric,
        "diagnostic_gate": diagnostic_gate,
        "decision": (
            "存在跨被试闭集行级信号"
            if diagnostic_gate
            else "未建立跨被试闭集行级信号"
        ),
        "metrics": aggregate_metrics,
        "paired_bootstrap": paired,
        "folds": fold_summaries,
        "queries": all_records,
    }
    if save:
        output_dir = Path(config["closed_set_diagnostic"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        save_summary(output_dir / "diagnostic_summary.json", summary)
    return summary, top_ks


def run_training(config, smoke=False, save=True):
    """完成训练、验证和两个不触碰测试集的对照。"""
    training_config = dict(config["training"])
    if smoke:
        training_config["epochs"] = 1
        training_config["patience"] = 1
        training_config["batch_size"] = min(int(training_config["batch_size"]), 8)

    seed = int(training_config["seed"])
    set_seed(seed)
    device = choose_device(training_config.get("device", "auto"))
    dataset_module, train_dataset, val_dataset = build_datasets(config, smoke=smoke)
    mean, std = estimate_channel_stats(train_dataset)
    train_dataset.set_normalization(mean, std)
    val_dataset.set_normalization(mean, std)

    train_loader = make_loader(train_dataset, training_config, shuffle=True)
    val_loader = make_loader(val_dataset, training_config, shuffle=False)
    model_config = dict(config.get("model", {}))
    model_config.setdefault("embedding_dimension", config["dataset"]["embedding_dimension"])
    model = build_strided_conv_encoder(
        len(train_dataset.channel_names),
        train_dataset.channel_positions,
        train_dataset.subject_count,
        model_config,
    ).to(device)
    optimizer = build_optimizer(model, training_config)
    total_steps = max(len(train_loader) * int(training_config["epochs"]), 1)
    scheduler = build_scheduler(
        optimizer,
        total_steps,
        warmup_ratio=float(training_config.get("warmup_ratio", 0.05)),
    )
    use_amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    top_ks = tuple(int(k) for k in training_config.get("top_ks", [1, 5, 10]))
    selection_metric = f"macro_recall_at_{max(top_ks)}"

    output_dir = Path(training_config["output_dir"])
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(int(training_config["epochs"])):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, training_config
        )
        val_metrics = evaluate_model(model, val_loader, device, top_ks)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})
        score = val_metrics[selection_metric]
        if epoch > 0:
            print()
        print(f"第 {epoch + 1} 轮")
        print(f"  训练损失：{train_loss:.4f}")
        print(
            f"  验证宏平均 R@{max(top_ks)}："
            f"{format_percentage(score)}"
        )

        if score > best_score:
            best_score = score
            best_state = cpu_state_dict(model)
            epochs_without_improvement = 0
            if save:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    mean,
                    std,
                    train_dataset.channel_names,
                    train_dataset.channel_positions,
                    train_dataset.subject_to_index,
                    model_config,
                    val_metrics,
                )
        else:
            epochs_without_improvement += 1
        if save:
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                mean,
                std,
                train_dataset.channel_names,
                train_dataset.channel_positions,
                train_dataset.subject_to_index,
                model_config,
                val_metrics,
            )
        if epochs_without_improvement >= int(training_config["patience"]):
            break

    if best_state is None:
        raise RuntimeError("训练没有产生可用 checkpoint。")
    model.load_state_dict(best_state)
    wrong_dataset, zero_dataset = build_control_datasets(
        dataset_module, val_dataset, mean, std
    )
    clean_metrics = evaluate_model(model, val_loader, device, top_ks)
    wrong_metrics = evaluate_model(
        model, make_loader(wrong_dataset, training_config, shuffle=False), device, top_ks
    )
    zero_metrics = evaluate_model(
        model, make_loader(zero_dataset, training_config, shuffle=False), device, top_ks
    )
    summary = {
        "status": "smoke_completed" if smoke else "completed",
        "device": str(device),
        "test_eeg_opened": False,
        "train_rows": len(train_dataset),
        "validation_rows": len(val_dataset),
        "eeg_shape": [len(train_dataset.channel_names), train_dataset.window_samples],
        "subject_count": train_dataset.subject_count,
        "subject_to_index": train_dataset.subject_to_index,
        "model": model_config,
        "selection_metric": selection_metric,
        "best_score": best_score,
        "history": history,
        "validation": {
            "candidate_definition": "验证集中可训练行的唯一文本ID",
            "clean": clean_metrics,
            "same_recording_wrong_row": wrong_metrics,
            "zero_eeg": zero_metrics,
        },
    }
    if save:
        save_summary(output_dir / "training_summary.json", summary)
    return summary


def parse_args(argv=None):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--prepare", action="store_true", help="构造并保存行级事件表")
    parser.add_argument("--smoke", action="store_true", help="只跑少量样本和一个 epoch")
    parser.add_argument("--no-save", action="store_true", help="不写 checkpoint 和摘要")
    parser.add_argument("--evaluate-checkpoint", help="只评估已有 checkpoint，不重新训练")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument(
        "--closed-set-loso",
        action="store_true",
        help="运行训练run内的嵌套留出被试闭集诊断",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """训练命令行入口。"""
    args = parse_args(argv)
    config = load_config(args.config)
    if args.prepare:
        print(prepare_event_table(config))
        return
    if args.closed_set_loso:
        summary, top_ks = run_closed_set_loso(
            config, smoke=args.smoke, save=not args.no_save
        )
        print_closed_set_summary(summary, top_ks)
        return
    if args.evaluate_checkpoint:
        analysis, top_ks = evaluate_checkpoint(
            config,
            args.evaluate_checkpoint,
            repetitions=args.bootstrap_repetitions,
            save=not args.no_save,
            smoke=args.smoke,
        )
        print_paired_analysis(analysis, top_ks)
        return
    summary = run_training(config, smoke=args.smoke, save=not args.no_save)
    print_training_summary(summary, tuple(config["training"]["top_ks"]))


if __name__ == "__main__":
    main()
