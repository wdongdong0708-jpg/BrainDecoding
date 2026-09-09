"""审计三个数据集的一秒上下文检查点，仅使用训练元数据和验证脑信号。"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tasks.word_decoding.ChineseEEG2_LittlePrince import train as training
from tasks.word_decoding.ChineseEEG2_LittlePrince.evaluate import 验证检查点合同
from datasets import ChineseEEG2 as data
from metrics import retrieval_ranks, summarize_retrieval, normalize_rows
from models import build_brain_embedding_model


CHINESEEEG2_AUDIT_CONFIGS = {
    "ChineseEEG2": "configs/ChineseEEG2_LittlePrince_sub01_actual_reading_1s_cnn_warm_start.yaml",
    "ChineseEEG2ActualReading": "configs/ChineseEEG2_LittlePrince_sub01_actual_reading_1s_cnn_warm_start.yaml",
    "ChineseEEG2ActualReadingTwoSubjects": "configs/ChineseEEG2_LittlePrince_sub01_sub02_actual_reading_1s_cnn_warm_start.yaml",
    "ChineseEEG2ActualReadingFourSubjects": "configs/ChineseEEG2_LittlePrince_sub01_sub04_actual_reading_1s_cnn_warm_start.yaml",
    "ChineseEEG2ActualReadingFourSubjectsRandomInit": "configs/ChineseEEG2_LittlePrince_sub01_sub04_actual_reading_1s.yaml",
}


def 位置特征(table):
    """只使用组内序号和有效组长度，不使用词、章节或记录身份。"""
    group = table.groupby("sentence_uid", sort=False)
    position = group.cumcount().to_numpy()
    length = group.event_id.transform("size").to_numpy()
    relative = np.floor(4 * position / np.maximum(length - 1, 1)).astype(int)
    return np.column_stack([position, length, relative,
                            np.asarray([f"{n}:{p}" for n, p in zip(length, position)])]).astype(str)


def 分数排名(scores, words, candidates):
    """与主指标一致，目标得分并列时取中间名次。"""
    index = {word: i for i, word in enumerate(candidates)}
    target = scores[np.arange(len(words)), [index[w] for w in words]][:, None]
    return 1 + (scores > target).sum(1) + 0.5 * ((scores == target).sum(1) - 1)


@torch.inference_mode()
def 上下文预测(model, features, batches, groups, permutation, device):
    """只错配逐词 CNN 输出，保持组位置、中心目标和批次不变。"""
    predictions = []
    for indices in batches:
        x = features[permutation[indices]].to(device)
        g = torch.as_tensor(groups[indices], device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            out = model.context_transformer(x, g)
            out = torch.nn.functional.normalize(out, dim=-1)
        predictions.append(out.float().cpu())
    return torch.cat(predictions).numpy()


def 记录替换审计(model, features, batches, groups, val, vocabulary, embedding_map, clean, device, checkpoint_path):
    """固定共同可替换且支持充足的查询，比较记录内外供体与预测稳定性。"""
    rows = [np.asarray(x) for x in val.groupby("sentence_uid", sort=False).indices.values()]
    records = [str(val.iloc[x[0]].recording_id) for x in rows]
    donors = {"same_record": {}, "cross_record": {}}
    if val.recording_id.nunique() < 2:
        donors.pop("cross_record")
    eligible = np.zeros(len(val), dtype=bool)
    for i, indices in enumerate(rows):
        for condition in donors:
            donors[condition][i] = [j for j, other in enumerate(rows)
                if len(other) == len(indices) and j != i
                and ((records[j] == records[i]) == (condition == "same_record"))]
        if all(donors[c][i] for c in donors):
            eligible[indices] = True
    words = val.normalized_word.astype(str).to_numpy()
    observed = list(dict.fromkeys(words[np.isin(words, vocabulary)]))
    support = pd.Series(words[eligible & np.isin(words, vocabulary)]).value_counts()
    retained = support[support >= 10].index.tolist()
    selected = np.flatnonzero(eligible & np.isin(words, retained))
    assert len(selected) > 0
    targets = words[selected]
    embeddings = normalize_rows(np.stack([embedding_map[w] for w in observed]))
    original = normalize_rows(clean[selected]) @ embeddings.T
    order = np.argsort(-original, axis=1, kind="stable")
    original_ranks = 分数排名(original, targets, observed)
    output = checkpoint_path.parent / "audit_record_matched_v1"
    output.mkdir(exist_ok=True)
    protocol = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=training.文件摘要(checkpoint_path),
        test_eeg_opened=False, training_eeg_opened=False, seeds=list(range(20)),
        minimum_support_after_joint_eligibility=10, candidate_words=observed,
        retained_words=retained, query_count=len(selected),
        support={w: int(support[w]) for w in retained},
        donor_sampling="uniform_group_with_replacement_no_target_label_matching",
        unavailable_controls=["cross_record"] if "cross_record" not in donors else [],
        clean=summarize_retrieval(original_ranks, targets, len(observed)), controls={})
    training.save_json(output / "protocol.json", protocol)
    print("固定共同查询", len(selected), "目标词", len(retained), "正确输入", protocol["clean"], flush=True)
    query = val.iloc[selected][["event_id", "recording_id", "sentence_uid", "normalized_word"]].copy()
    query["clean_rank"] = original_ranks
    query["clean_top1"] = np.asarray(observed)[order[:, 0]]
    arrays = dict(selected=selected, clean_scores=original, clean_ranks=original_ranks)
    for condition in donors:
        scores_all, donor_all, runs = [], [], []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            permutation = np.arange(len(val))
            for i, indices in enumerate(rows):
                if eligible[indices[0]]:
                    permutation[indices] = rows[int(rng.choice(donors[condition][i]))]
            assert np.all(permutation[selected] != selected)
            moved_records = val.recording_id.to_numpy()[permutation[selected]]
            assert np.all((moved_records == val.recording_id.to_numpy()[selected]) == (condition == "same_record"))
            prediction = 上下文预测(model, features, batches, groups, permutation, device)
            scores = normalize_rows(prediction[selected]) @ embeddings.T
            ranks = 分数排名(scores, targets, observed)
            run = summarize_retrieval(ranks, targets, len(observed))
            run.update(seed=seed, changed_label_fraction=float((words[permutation[selected]] != targets).mean()))
            runs.append(run)
            scores_all.append(scores)
            donor_all.append(permutation[selected])
            if (seed + 1) % 5 == 0:
                print(condition, f"已完成 {seed + 1}/20", flush=True)
        scores_all = np.stack(scores_all)
        ordered = np.argsort(-scores_all, axis=2, kind="stable")
        same_top1 = ordered[:, :, 0] == order[None, :, 0]
        overlap = (ordered[:, :, :10, None] == order[None, :, None, :10]).any(axis=3).mean(axis=2)
        ranks_all = np.stack([分数排名(s, targets, observed) for s in scores_all])
        protocol["controls"][condition] = dict(runs=runs,
            macro_top10_mean=float(np.mean([r["macro_recall_at_10"] for r in runs])),
            macro_top1_mean=float(np.mean([r["macro_recall_at_1"] for r in runs])),
            top1_unchanged_fraction=float(same_top1.mean()), top10_overlap_fraction=float(overlap.mean()),
            top10_set_unchanged_fraction=float((overlap == 1).mean()),
            target_rank_absolute_change=float(np.abs(ranks_all - original_ranks).mean()),
            target_rank_unchanged_fraction=float((ranks_all == original_ranks).mean()))
        for name, values in {"top1_unchanged": same_top1, "top10_overlap": overlap,
                             "target_rank": ranks_all, "top10_hit": ranks_all <= 10}.items():
            query[condition + "_" + name] = values.mean(axis=0)
        arrays[condition + "_scores"] = scores_all
        arrays[condition + "_donor_indices"] = np.stack(donor_all)
        print(condition, {k: v for k, v in protocol["controls"][condition].items() if k != "runs"}, flush=True)
    query.to_csv(output / "逐查询稳定性.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(output / "预测与供体.npz", **arrays, event_ids=val.event_id.to_numpy(dtype=str), candidates=np.asarray(observed))
    training.save_json(output / "audit_summary.json", protocol)


def main(dataset_name="ChineseEEG2", record_audit=False):
    """复评、错配对照、训练集拟合结构基线，并保存逐词证据。"""
    if dataset_name in CHINESEEEG2_AUDIT_CONFIGS:
        task_training, module = training, data
        config_path = CHINESEEEG2_AUDIT_CONFIGS[dataset_name]
        config = training.载入配置(config_path)
        event_path = training.确保事件表(config)
        train = data.载入事件表(event_path, split="train")
        val = data.载入事件表(event_path, split="val")
        vocabulary = data.载入候选词(event_path)
        dataset = training.构建数据集(config, val)
        signal_key = "eeg"
    else:
        if dataset_name in ("LibriBrain100", "LibriBrain100WarmStart"):
            from tasks.word_decoding.LibriBrain100 import train as task_training
            from datasets import LibriBrain as module
            config_path = (
                "configs/LibriBrain100_1s_cnn_warm_start.yaml"
                if dataset_name == "LibriBrain100WarmStart"
                else "configs/LibriBrain100_1s.yaml"
            )
            vocabulary = module.LIBRIBRAIN100_50_WORD_VOCABULARY
        elif dataset_name == "SMN4Lang":
            from tasks.word_decoding.SMN4Lang import train as task_training
            from datasets import SMN4Lang as module
            config_path = "configs/SMN4Lang_1s_cnn_warm_start.yaml"
            vocabulary = module.SMN4LANG50_VOCABULARY
        else:
            raise ValueError(f"未知数据集：{dataset_name}")
        config = task_training.load_config(config_path)
        event_path = task_training.ensure_event_table(config)
        train = module.load_event_table(event_path, split="train")
        val = module.load_event_table(event_path, split="val")
        dataset = task_training.build_dataset(config, val)
        signal_key = "meg"
    checkpoint_path = Path(config["training"]["output_dir"]) / "best.pt"
    output = checkpoint_path.parent / "audit_v1"
    output.mkdir(exist_ok=True)
    training.set_seed(0)
    device = training.choose_device("cuda")
    assert not set(train.event_id) & set(val.event_id)
    assert not set(train.recording_id) & set(val.recording_id)
    loader, _ = task_training.make_loader(dataset, config["training"], False)
    batches = [np.asarray(indices) for indices in loader.batch_sampler]
    sampled_order = np.concatenate(batches)
    if len(sampled_order) != len(val) or len(np.unique(sampled_order)) != len(val):
        raise ValueError("验证采样器没有恰好覆盖每个事件一次。")
    val = val.iloc[sampled_order].reset_index(drop=True)
    ordered_batches = []
    cursor = 0
    for indices in batches:
        ordered_batches.append(np.arange(cursor, cursor + len(indices)))
        cursor += len(indices)
    batches = ordered_batches
    checkpoint = training.load_checkpoint(checkpoint_path, map_location="cpu")
    if dataset_name in CHINESEEEG2_AUDIT_CONFIGS:
        验证检查点合同(checkpoint, config, dataset, vocabulary)
    else:
        canonical_dataset_name = (
            "LibriBrain100"
            if dataset_name == "LibriBrain100WarmStart"
            else dataset_name
        )
        assert checkpoint["task"] == f"word_decoding/{canonical_dataset_name}"
        assert tuple(checkpoint["channel_names"]) == tuple(dataset.channel_names)
        if checkpoint.get("text_embedding_config") is not None:
            assert checkpoint["text_embedding_config"] == config["text_embedding"]
    model = build_brain_embedding_model(dataset.channel_count, dataset.channel_positions,
                                       dataset.subject_count, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    # 先通过原入口复评，再核对逐词特征重组前向的数值。
    if dataset_name in CHINESEEEG2_AUDIT_CONFIGS:
        reference_metrics, reference = training.评估数据(model, loader, device, vocabulary)
    else:
        reference_metrics, reference = task_training.evaluate_loader(model, loader, device)
    print(dataset_name, "原入口复评", reference_metrics, flush=True)
    pieces = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                features = model.brain_encoder(batch[signal_key].to(device), batch["subject_index"].to(device))
                pieces.append(torch.nn.functional.normalize(features, dim=-1).cpu())
    features = torch.cat(pieces)
    groups = dataset.sentence_indices[sampled_order]
    identity = np.arange(len(val))
    clean = 上下文预测(model, features, batches, groups, identity, device)
    np.testing.assert_allclose(clean, reference["predictions"].numpy(), rtol=1e-5, atol=1e-6)
    if record_audit:
        记录替换审计(model, features, batches, groups, val, vocabulary, dataset.embedding_map,
                     clean, device, checkpoint_path)
        return
    words = val.normalized_word.astype(str).to_numpy()
    selected = np.flatnonzero(np.isin(words, vocabulary))
    targets = words[selected]
    candidates = list(dict.fromkeys(targets))
    embeddings = np.stack([dataset.embedding_map[w] for w in candidates])
    clean_ranks = retrieval_ranks(clean[selected], targets, embeddings, candidates)
    clean_metrics = summarize_retrieval(clean_ranks, targets, len(candidates))
    assert abs(clean_metrics["macro_recall_at_10"] - reference_metrics["macro_recall_at_10"]) < 1e-10
    saved_summary = json.loads((checkpoint_path.parent / "training_summary.json").read_text(encoding="utf-8"))
    assert abs(clean_metrics["macro_recall_at_10"] - saved_summary["best_score"]) < 1e-10
    result = dict(dataset=dataset_name, checkpoint=str(checkpoint_path), checkpoint_sha256=training.文件摘要(checkpoint_path),
                  checkpoint_epoch=int(checkpoint["epoch"]), test_neural_data_opened=False, training_neural_data_opened=False,
                  test_eeg_opened=False, training_eeg_opened=False,
                  selection_status="previously_selected_validation_checkpoint",
                  clean=clean_metrics, controls={}, metadata_baselines={})
    full_embeddings = np.stack([dataset.embedding_map[w] for w in vocabulary])
    full_ranks = retrieval_ranks(clean[selected], targets, full_embeddings, list(vocabulary))
    result["fixed_50_candidates"] = summarize_retrieval(full_ranks, targets, 50)
    row_groups = [np.asarray(indices) for indices in val.groupby("sentence_uid", sort=False).indices.values()]
    buckets = {}
    for indices in row_groups:
        buckets.setdefault((str(val.iloc[indices[0]].recording_id), len(indices)), []).append(indices)
    rank_arrays = {"clean": clean_ranks, "full50": full_ranks}
    for condition in ("within_row", "cross_row_same_length"):
        runs = []
        ranks_all = []
        for seed in range(20):
            random = np.random.default_rng(seed)
            permutation = identity.copy()
            if condition == "within_row":
                for indices in row_groups:
                    if len(indices) > 1:
                        permutation[indices] = np.roll(indices, int(random.integers(1, len(indices))))
            else:
                for rows in buckets.values():
                    if len(rows) > 1:
                        order = random.permutation(len(rows))
                        for a, b in zip(order, np.roll(order, 1)):
                            permutation[rows[a]] = rows[b]
            moved = permutation != identity
            if condition == "cross_row_same_length":
                assert all(val.iloc[i].recording_id == val.iloc[permutation[i]].recording_id for i in selected)
                assert all(val.iloc[i].sentence_uid != val.iloc[permutation[i]].sentence_uid for i in selected if moved[i])
            prediction = 上下文预测(model, features, batches, groups, permutation, device)
            ranks = retrieval_ranks(prediction[selected], targets, embeddings, candidates)
            metrics = summarize_retrieval(ranks, targets, len(candidates))
            metrics.update(seed=seed, moved_query_fraction=float(moved[selected].mean()),
                           changed_label_fraction=float((words[permutation[selected]] != targets).mean()))
            eligible = moved[selected]
            metrics["moved_only_clean"] = summarize_retrieval(clean_ranks[eligible], targets[eligible], len(candidates))
            metrics["moved_only_control"] = summarize_retrieval(ranks[eligible], targets[eligible], len(candidates))
            runs.append(metrics)
            ranks_all.append(ranks)
            if (seed + 1) % 5 == 0:
                print(dataset_name, condition, f"已完成 {seed + 1}/20", flush=True)
        rank_arrays[condition] = np.stack(ranks_all)
        result["controls"][condition] = dict(
            runs=runs, macro_top10_mean=float(np.mean([m["macro_recall_at_10"] for m in runs])),
            macro_top10_min=float(min(m["macro_recall_at_10"] for m in runs)),
            macro_top10_max=float(max(m["macro_recall_at_10"] for m in runs)),
            macro_top1_mean=float(np.mean([m["macro_recall_at_1"] for m in runs])))
        print(condition, result["controls"][condition]["macro_top10_mean"], flush=True)
    # 所有基线仅从训练词表内样本拟合，不以验证标签调参。
    train_selected = train.normalized_word.isin(vocabulary).to_numpy()
    encoder = OneHotEncoder(handle_unknown="ignore")
    train_x = encoder.fit_transform(位置特征(train)[train_selected])
    val_x = encoder.transform(位置特征(val)[selected])
    train_y = train.loc[train_selected, "normalized_word"].astype(str).to_numpy()
    frequency = pd.Series(train_y).value_counts()
    baseline_scores = {"training_frequency": np.tile([frequency.get(w, 0) for w in candidates], (len(selected), 1))}
    for weight in (None, "balanced"):
        estimator = LogisticRegression(C=1.0, max_iter=2000, class_weight=weight, solver="lbfgs")
        estimator.fit(train_x, train_y)
        probability = estimator.predict_proba(val_x)
        lookup = {w: i for i, w in enumerate(estimator.classes_)}
        baseline_scores["position_length_" + str(weight)] = probability[:, [lookup[w] for w in candidates]]
    for name, scores in baseline_scores.items():
        ranks = 分数排名(scores, targets, candidates)
        result["metadata_baselines"][name] = summarize_retrieval(ranks, targets, len(candidates))
        rank_arrays[name] = ranks
        print(name, result["metadata_baselines"][name], flush=True)
    query = val.iloc[selected][[c for c in ("event_id", "recording_id", "sentence_uid", "normalized_word", "material_line") if c in val]].copy()
    query["rank"] = clean_ranks
    similarity = normalize_rows(clean[selected]) @ normalize_rows(embeddings).T
    ordered = np.argsort(-similarity, axis=1, kind="stable")
    query["predicted_top1"] = np.asarray(candidates)[ordered[:, 0]]
    query["predicted_top10"] = [" / ".join(np.asarray(candidates)[indices[:10]]) for indices in ordered]
    for name in ("within_row", "cross_row_same_length"):
        query[name + "_top10_mean"] = (rank_arrays[name] <= 10).mean(axis=0)
    query.to_csv(output / "逐查询排名.csv", index=False, encoding="utf-8-sig")
    rows = []
    for word in candidates:
        mask = targets == word
        rows.append(dict(word=word, support=int(mask.sum()), top1=float((clean_ranks[mask] <= 1).mean()),
                         top10=float((clean_ranks[mask] <= 10).mean()), median_rank=float(np.median(clean_ranks[mask])),
                         within_row=float((rank_arrays["within_row"][:, mask] <= 10).mean()),
                         cross_row=float((rank_arrays["cross_row_same_length"][:, mask] <= 10).mean())))
    per_word = pd.DataFrame(rows).sort_values(["top10", "support"], ascending=False)
    per_word.to_csv(output / "逐词贡献.csv", index=False, encoding="utf-8-sig")
    result["support_sensitivity"] = {}
    for threshold in (1, 3, 5, 10, 20):
        keep = per_word[per_word.support >= threshold]
        result["support_sensitivity"][str(threshold)] = dict(words=len(keep), queries=int(keep.support.sum()),
            macro_top10=float(keep.top10.mean()), macro_top1=float(keep.top1.mean()))
    result["rank_bins"] = {name: int(mask.sum()) for name, mask in {
        "rank1": clean_ranks <= 1, "rank2_5": (clean_ranks > 1) & (clean_ranks <= 5),
        "rank6_10": (clean_ranks > 5) & (clean_ranks <= 10), "rank11plus": clean_ranks > 10}.items()}
    result["most_predicted_top1"] = query.predicted_top1.value_counts().head(10).to_dict()
    result["support_counts"] = per_word.support.describe().to_dict()
    result["by_recording"] = {}
    recordings = val.iloc[selected].recording_id.to_numpy()
    for recording in dict.fromkeys(recordings):
        mask = recordings == recording
        result["by_recording"][str(recording)] = {
            "clean": summarize_retrieval(clean_ranks[mask], targets[mask], len(candidates)),
            "cross_row_top10_mean": float(np.mean([
                summarize_retrieval(r[mask], targets[mask], len(candidates))["macro_recall_at_10"]
                for r in rank_arrays["cross_row_same_length"]])),
        }
    np.savez_compressed(output / "审计排名.npz", **rank_arrays, targets=targets,
                        event_ids=val.iloc[selected].event_id.to_numpy(dtype=str))
    training.save_json(output / "audit_summary.json", result)
    print("完成", output, flush=True)


def 错词分析(dataset_name="ChineseEEG2"):
    """复用固定查询的候选分数，分析错误方向及供体跟随，不读取脑信号。"""
    if dataset_name in CHINESEEEG2_AUDIT_CONFIGS:
        config = training.载入配置(CHINESEEEG2_AUDIT_CONFIGS[dataset_name])
        load_table = data.载入事件表
    elif dataset_name in ("LibriBrain100", "LibriBrain100WarmStart"):
        from tasks.word_decoding.LibriBrain100 import train as task_training
        from datasets import LibriBrain as module
        config = task_training.load_config(
            "configs/LibriBrain100_1s_cnn_warm_start.yaml"
            if dataset_name == "LibriBrain100WarmStart"
            else "configs/LibriBrain100_1s.yaml"
        )
        load_table = module.load_event_table
        vocabulary = module.LIBRIBRAIN100_50_WORD_VOCABULARY
    elif dataset_name == "SMN4Lang":
        from tasks.word_decoding.SMN4Lang import train as task_training
        from datasets import SMN4Lang as module
        config = task_training.load_config("configs/SMN4Lang_1s_cnn_warm_start.yaml")
        load_table = module.load_event_table
        vocabulary = module.SMN4LANG50_VOCABULARY
    else:
        raise ValueError(f"未知数据集：{dataset_name}")
    event_path = Path(config["cache"]["event_table"])
    train = load_table(event_path, split="train")
    val = load_table(event_path, split="val")
    folder = Path(config["training"]["output_dir"]) / "audit_record_matched_v1"
    saved = np.load(folder / "预测与供体.npz")
    saved_event_ids = saved["event_ids"].astype(str)
    row_by_event = {
        event_id: index for index, event_id in enumerate(val.event_id.astype(str))
    }
    if len(row_by_event) != len(val) or any(event_id not in row_by_event for event_id in saved_event_ids):
        raise ValueError("记录替换审计的事件集合与当前验证事件表不一致。")
    val = val.iloc[[row_by_event[event_id] for event_id in saved_event_ids]].reset_index(drop=True)
    assert np.array_equal(saved["event_ids"], val.event_id.to_numpy(dtype=str))
    selected = saved["selected"]
    candidates = saved["candidates"].tolist()
    lookup = {w: i for i, w in enumerate(candidates)}
    words = val.normalized_word.astype(str).to_numpy()
    targets = words[selected]
    scores = saved["clean_scores"]
    pred = scores.argmax(1)
    target = np.array([lookup[w] for w in targets])
    error = pred != target
    freq = train.normalized_word.value_counts().reindex(candidates, fill_value=0).to_numpy()
    length = np.array([len(w) for w in candidates])
    duration = train.word_duration_seconds if "word_duration_seconds" in train else train.aligned_stop_seconds - train.aligned_start_seconds
    typical = duration.groupby(train.normalized_word).median().reindex(candidates).to_numpy()
    actual = (val.word_duration_seconds if "word_duration_seconds" in val else val.aligned_stop_seconds - val.aligned_start_seconds).to_numpy()[selected]
    embeddings = data.load_text_embedding_cache(config["cache"]["text_embeddings"])
    text = normalize_rows(np.stack([embeddings[w] for w in candidates]))
    similarity = text @ text.T
    table = val.iloc[selected][["event_id", "recording_id", "sentence_uid", "normalized_word"]].copy()
    table["predicted_word"] = np.asarray(candidates)[pred]
    table["correct"] = ~error
    table["target_duration"] = actual
    table["predicted_training_frequency"] = freq[pred]
    table["same_character_length"] = length[target] == length[pred]
    table["embedding_similarity"] = similarity[target, pred]
    positions = 位置特征(val)[selected]
    table["position"] = positions[:, 0]
    table["group_length"] = positions[:, 1]
    table["relative_position_bin"] = positions[:, 2]
    table.to_csv(folder / "错词逐查询.csv", index=False, encoding="utf-8-sig")
    confusion = table[error].groupby(["normalized_word", "predicted_word"]).size().rename("count").reset_index()
    confusion["target_support"] = confusion.normalized_word.map(pd.Series(targets).value_counts())
    confusion.sort_values("count", ascending=False).to_csv(folder / "错误流向.csv", index=False, encoding="utf-8-sig")
    def 关联指标(indices):
        return dict(same_length=float((length[target[error]] == length[indices]).mean()),
            duration_log_distance=float(np.abs(np.log(np.maximum(actual[error], .001)) - np.log(np.maximum(typical[indices], .001))).mean()),
            text_similarity=float(similarity[target[error], indices].mean()))
    result = dict(query_count=len(selected), error_count=int(error.sum()), neural_data_opened=False,
        top_errors=confusion.sort_values("count", ascending=False).head(20).to_dict("records"),
        predicted_error_counts=pd.Series(np.asarray(candidates)[pred[error]]).value_counts().to_dict(),
        training_frequency={w: int(freq[i]) for i, w in enumerate(candidates)},
        prediction_more_frequent_than_target=float((freq[pred[error]] > freq[target[error]]).mean()),
        observed_error_associations=关联指标(pred[error]), permutation_controls={}, donor_following={})
    from sklearn.metrics import mutual_info_score
    result["frequency_spearman"] = float(pd.Series(freq).corr(
        pd.Series(np.bincount(pred[error], minlength=len(candidates))), method="spearman"))
    result["error_prediction_mutual_information"] = {name: float(mutual_info_score(table.loc[error, name], pred[error]))
        for name in ("recording_id", "relative_position_bin", "group_length")}
    result["structure_baselines"] = {}
    if dataset_name in CHINESEEEG2_AUDIT_CONFIGS:
        vocabulary = data.载入候选词(event_path)
    train_mask = train.normalized_word.isin(vocabulary).to_numpy()
    train_features = 位置特征(train)[train_mask]
    train_targets = train.normalized_word.to_numpy()[train_mask]
    for name, columns in {"row_length_only": [1], "absolute_position_only": [0],
                           "relative_position_only": [2], "length_and_positions": [0, 1, 2, 3]}.items():
        encoder = OneHotEncoder(handle_unknown="ignore")
        x = encoder.fit_transform(train_features[:, columns])
        model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        model.fit(x, train_targets)
        probability = model.predict_proba(encoder.transform(positions[:, columns]))
        mapping = {w: i for i, w in enumerate(model.classes_)}
        ordered = probability[:, [mapping[w] for w in candidates]]
        result["structure_baselines"][name] = summarize_retrieval(分数排名(ordered, targets, candidates), targets, len(candidates))
    result["error_predictions_by_structure"] = {}
    for name in ("group_length", "relative_position_bin"):
        result["error_predictions_by_structure"][name] = {
            str(key): dict(count=len(frame), top_predictions=frame.predicted_word.value_counts().head(4).to_dict())
            for key, frame in table[error].groupby(name) if len(frame) >= 10}
    # 保留实际错误预测的词频边际，逐步保留记录、真词长度和位置，比较关联是否仍存在。
    error_indices = np.flatnonzero(error)
    for condition in ("global", "record", "record_length_position", "record_target_absolute_position"):
        keys = ["all"] * len(error_indices) if condition == "global" else [
            str(table.iloc[i].recording_id) + (f"|{length[target[i]]}|{positions[i, 2]}" if condition == "record_length_position" else "")
            for i in error_indices]
        if condition == "record_target_absolute_position":
            keys = [f"{table.iloc[i].recording_id}|{targets[i]}|{positions[i, 0]}" for i in error_indices]
        buckets = [np.flatnonzero(np.asarray(keys) == key) for key in dict.fromkeys(keys)]
        results = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            shuffled = pred[error].copy()
            for bucket in buckets:
                # 交换时禁止产生正确配对，避免对照混入真词自身的嵌入相似度。
                for a, b in rng.choice(bucket, size=(10 * len(bucket), 2)):
                    if shuffled[b] != target[error_indices[a]] and shuffled[a] != target[error_indices[b]]:
                        shuffled[a], shuffled[b] = shuffled[b], shuffled[a]
            results.append(关联指标(shuffled))
            results[-1].update({"mi_" + name: float(mutual_info_score(table.loc[error, name], shuffled))
                for name in ("recording_id", "relative_position_bin", "group_length")})
        result["permutation_controls"][condition] = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
        result["permutation_controls"][condition]["non_singleton_fraction"] = float(sum(len(b) for b in buckets if len(b) > 1) / len(error_indices))
    # 记录词频仅用于事后解释，并非训练输入或合法的无标签解码基线。
    record_probs = {}
    for record, frame in val.groupby("recording_id"):
        counts = frame.normalized_word.value_counts().reindex(candidates, fill_value=0).to_numpy() + 1
        record_probs[record] = counts / counts.sum()
    for condition in ("same_record", "cross_record"):
        if condition + "_scores" not in saved:
            continue
        runs = []
        for seed, changed in enumerate(saved[condition + "_scores"]):
            donors = saved[condition + "_donor_indices"][seed]
            donor = np.array([lookup.get(w, -1) for w in words[donors]])
            mask = (donor >= 0) & (donor != target)
            ii = np.flatnonzero(mask)
            dd = donor[mask]
            clean_rank = 1 + (scores[ii] > scores[ii, dd, None]).sum(1)
            changed_rank = 1 + (changed[ii] > changed[ii, dd, None]).sum(1)
            changed_pred = changed.argmax(1)
            run = dict(eligible=int(mask.sum()), clean_donor_top1=float((pred[ii] == dd).mean()),
                replaced_donor_top1=float((changed_pred[ii] == dd).mean()),
                clean_donor_top10=float((clean_rank <= 10).mean()), replaced_donor_top10=float((changed_rank <= 10).mean()),
                donor_rank_improvement=float((clean_rank - changed_rank).mean()))
            donor_record = val.recording_id.to_numpy()[donors]
            target_record = val.recording_id.to_numpy()[selected]
            logratio = np.stack([np.log(record_probs[d] / record_probs[t]) for d, t in zip(donor_record, target_record)])
            run["donor_vs_target_record_log_preference_shift"] = float((logratio[np.arange(len(pred)), changed_pred] - logratio[np.arange(len(pred)), pred]).mean())
            runs.append(run)
        result["donor_following"][condition] = dict(runs=runs, mean={k: float(np.mean([r[k] for r in runs])) for k in runs[0]})
    training.save_json(folder / "error_analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    if "--errors" in sys.argv:
        错词分析(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "ChineseEEG2")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "ChineseEEG2", "--record-matched" in sys.argv)
