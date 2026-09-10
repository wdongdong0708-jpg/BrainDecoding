"""词级训练入口共同使用的批处理与编码函数。"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


class SentenceBatchSampler(Sampler):
    """打乱序列组，同时严格限制实际批次大小。

    超过 ``batch_size`` 的组会被切成连续块。这样既保留块内顺序，也不会让
    较长的记录组或故事组在无提示的情况下形成超大批次。
    """

    def __init__(self, sentence_uids, batch_size, shuffle, seed):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        groups = {}
        for index, value in enumerate(sentence_uids):
            groups.setdefault(str(value), []).append(index)
        self.groups = list(groups.values())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _ordered_groups(self):
        order = np.arange(len(self.groups))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        return [self.groups[index] for index in order]

    def __iter__(self):
        batch = []
        for group in self._ordered_groups():
            chunks = [
                group[start : start + self.batch_size]
                for start in range(0, len(group), self.batch_size)
            ]
            for chunk in chunks:
                if batch and len(batch) + len(chunk) > self.batch_size:
                    yield batch
                    batch = []
                if len(chunk) == self.batch_size:
                    if batch:
                        yield batch
                        batch = []
                    yield chunk
                else:
                    batch.extend(chunk)
        if batch:
            yield batch

    def __len__(self):
        count = 0
        batch_size = 0
        for group in self.groups:
            chunks = [
                group[start : start + self.batch_size]
                for start in range(0, len(group), self.batch_size)
            ]
            for chunk in chunks:
                if batch_size and batch_size + len(chunk) > self.batch_size:
                    count += 1
                    batch_size = 0
                if len(chunk) == self.batch_size:
                    if batch_size:
                        count += 1
                        batch_size = 0
                    count += 1
                else:
                    batch_size += len(chunk)
        return count + int(batch_size > 0)


def make_loader(dataset, training_config, shuffle):
    """按句组构建保持现有批次语义的 DataLoader。"""
    sampler = SentenceBatchSampler(
        dataset.table["sentence_uid"].astype(str).tolist(),
        batch_size=int(training_config["batch_size"]),
        shuffle=shuffle,
        seed=int(training_config["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def move_batch(batch, device):
    """移动当前两个 MEG 词级任务共同使用的批次字段。"""
    return (
        batch["meg"].to(device, non_blocking=True),
        batch["text_embedding"].to(device, non_blocking=True),
        batch["subject_index"].to(device, non_blocking=True),
        batch["sentence_index"].to(device, non_blocking=True),
    )


def train_one_epoch(
    model,
    loss_module,
    loader,
    optimizer,
    scaler,
    device,
    config,
    freeze_brain_encoder=False,
):
    """执行 LibriBrain100 与 SMN4Lang 已共同使用的 epoch 级训练。"""
    model.train()
    if freeze_brain_encoder:
        model.brain_encoder.eval()
    loss_module.train()
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
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
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach()) * len(meg)
        sample_count += len(meg)
    return loss_sum / max(sample_count, 1)


@torch.inference_mode()
def encode_loader(model, loader, device, amp=True):
    """编码完整数据划分并保留现有检索元数据。"""
    model.eval()
    predictions = []
    targets = []
    words = []
    event_ids = []
    recording_ids = []
    use_amp = bool(amp) and device.type == "cuda"
    for batch in loader:
        meg, text_embedding, subject_indices, sentence_indices = move_batch(
            batch, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            estimate = model(meg, subject_indices, sentence_indices)
        predictions.append(estimate.float().cpu())
        targets.append(text_embedding.float().cpu())
        words.extend(str(word) for word in batch["word"])
        event_ids.extend(str(value) for value in batch["event_id"])
        recording_ids.extend(str(value) for value in batch["recording_id"])
    return {
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "words": words,
        "event_ids": event_ids,
        "recording_ids": recording_ids,
    }
