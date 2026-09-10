"""跨任务共享的文本检索指标。"""

import numpy as np


def _as_numpy(values):
    """分离 PyTorch 张量，同时避免让 metrics.py 直接依赖 PyTorch。"""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def normalize_rows(values):
    """逐行 L2 归一化。"""
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def unique_candidates(text_embeddings, text_ids):
    """按文本 ID 去重候选，并检查重复目标向量是否一致。"""
    candidates = {}
    for embedding, text_id in zip(text_embeddings, text_ids):
        identifier = str(text_id)
        if identifier in candidates:
            if not np.allclose(candidates[identifier], embedding, atol=1e-5, rtol=1e-5):
                raise ValueError(f"同一文本 ID 对应了不同 BERT 向量：{identifier}")
        else:
            candidates[identifier] = np.asarray(embedding, dtype=np.float32)
    candidate_ids = list(candidates)
    candidate_matrix = np.stack([candidates[key] for key in candidate_ids])
    return candidate_matrix, candidate_ids


def retrieval_ranks(query_embeddings, target_ids, candidate_embeddings, candidate_ids, chunk_size=512):
    """按余弦相似度计算并列时取中间名次的检索排名。"""
    queries = normalize_rows(query_embeddings)
    candidates = normalize_rows(candidate_embeddings)
    candidate_index = {text_id: index for index, text_id in enumerate(candidate_ids)}
    target_indices = np.asarray([candidate_index[str(text_id)] for text_id in target_ids])
    ranks = np.empty(len(queries), dtype=np.float64)

    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        similarity = queries[start:stop] @ candidates.T
        local_targets = target_indices[start:stop]
        target_scores = similarity[np.arange(stop - start), local_targets][:, None]
        greater = (similarity > target_scores).sum(axis=1)
        equal = (similarity == target_scores).sum(axis=1)
        ranks[start:stop] = 1.0 + greater + 0.5 * (equal - 1)
    return ranks


def mean_by_target(values, target_ids):
    """先在同一文本内平均，再对文本做宏平均。"""
    _, inverse = np.unique(np.asarray(target_ids), return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    return float(np.mean(sums / counts))


def summarize_retrieval(ranks, target_ids, candidate_count, top_ks=(1, 5, 10)):
    """汇总宏/微 Recall 与排名指标。"""
    result = {
        "query_count": int(len(ranks)),
        "candidate_count": int(candidate_count),
        "target_count": int(len(np.unique(target_ids))),
        "median_rank": float(np.median(ranks)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
    }
    for k in top_ks:
        hits = ranks <= int(k)
        result[f"random_recall_at_{k}"] = min(float(k) / candidate_count, 1.0)
        result[f"micro_recall_at_{k}"] = float(np.mean(hits))
        result[f"macro_recall_at_{k}"] = mean_by_target(hits, target_ids)
    return result


def retrieval_metrics(query_embeddings, text_embeddings, text_ids, top_ks=(1, 5, 10)):
    """从查询和配对文本直接生成固定候选集检索指标。"""
    candidates, candidate_ids = unique_candidates(text_embeddings, text_ids)
    ranks = retrieval_ranks(query_embeddings, text_ids, candidates, candidate_ids)
    return summarize_retrieval(ranks, text_ids, len(candidate_ids), top_ks=top_ks)


def retrieval_metrics_with_ranks(query_embeddings, text_embeddings, text_ids, top_ks=(1, 5, 10)):
    """同时返回汇总指标与逐查询排名。"""
    candidates, candidate_ids = unique_candidates(text_embeddings, text_ids)
    ranks = retrieval_ranks(query_embeddings, text_ids, candidates, candidate_ids)
    metrics = summarize_retrieval(ranks, text_ids, len(candidate_ids), top_ks=top_ks)
    return metrics, ranks


def _fixed_vocabulary_inputs(
    query_embeddings,
    target_embeddings,
    target_words,
    vocabulary,
):
    """准备共享的固定词表查询、候选和原始行索引。"""
    queries = _as_numpy(query_embeddings).astype(np.float32, copy=False)
    targets = _as_numpy(target_embeddings).astype(np.float32, copy=False)
    words = np.asarray([str(word).lower() for word in target_words])
    vocabulary = {str(word).lower() for word in vocabulary}
    selected = np.flatnonzero(np.isin(words, list(vocabulary)))
    if len(selected) == 0:
        raise ValueError("评估 split 中没有固定词表里的查询词。")
    selected_queries = queries[selected]
    selected_targets = targets[selected]
    selected_words = words[selected]
    candidates, candidate_words = unique_candidates(selected_targets, selected_words)
    return (
        selected_queries,
        selected_words,
        candidates,
        candidate_words,
        selected,
        vocabulary,
    )


def fixed_vocabulary_retrieval(
    query_embeddings,
    target_embeddings,
    target_words,
    vocabulary,
    top_ks=(1, 10),
    vocabulary_name="fixed",
):
    """评估目标出现在固定词表中的查询，匹配词表时不区分大小写。

    候选集合为每个实际出现的词表词保留一个目标向量，宏平均准确率则让每个
    已出现词获得相同权重。这与 LibriBrain 基线回调一致，不会把同一词的重复
    实例视为不同候选。
    """
    (
        selected_queries,
        selected_words,
        candidates,
        candidate_words,
        selected,
        vocabulary,
    ) = _fixed_vocabulary_inputs(
        query_embeddings,
        target_embeddings,
        target_words,
        vocabulary,
    )
    ranks = retrieval_ranks(
        selected_queries,
        selected_words,
        candidates,
        candidate_words,
    )
    summary = summarize_retrieval(
        ranks,
        selected_words,
        len(candidate_words),
        top_ks=top_ks,
    )
    summary["vocabulary_size"] = int(len(vocabulary))
    summary["observed_vocabulary_size"] = int(len(candidate_words))
    summary["missing_vocabulary_words"] = sorted(vocabulary - set(candidate_words))
    vocabulary_name = str(vocabulary_name).strip()
    if not vocabulary_name or any(character.isspace() for character in vocabulary_name):
        raise ValueError("vocabulary_name must be a non-empty token without spaces.")
    for k in top_ks:
        summary[f"retrieval_acc{k}_vocab={vocabulary_name}"] = summary[
            f"micro_recall_at_{k}"
        ]
        summary[f"retrieval_acc{k}_vocab={vocabulary_name}_macro"] = summary[
            f"macro_recall_at_{k}"
        ]
    return summary, ranks, selected


def fixed_vocabulary_top1_predictions(
    query_embeddings,
    target_embeddings,
    target_words,
    vocabulary,
    chunk_size=512,
):
    """返回固定词表查询的真实词、Top-1 预测词和原始行索引。

    候选向量仍由 ``unique_candidates`` 从当前评价 split 的目标向量中
    产生，匹配规则也与既有固定词表指标相同。这个辅助函数只暴露余弦
    检索的 Top-1 输出，不改变排名或汇总指标的计算。
    """
    (
        selected_queries,
        selected_words,
        candidates,
        candidate_words,
        selected,
        _,
    ) = _fixed_vocabulary_inputs(
        query_embeddings,
        target_embeddings,
        target_words,
        vocabulary,
    )
    normalized_queries = normalize_rows(selected_queries)
    normalized_candidates = normalize_rows(candidates)
    candidate_words = np.asarray(candidate_words)
    predicted_words = []
    for start in range(0, len(normalized_queries), int(chunk_size)):
        stop = min(start + int(chunk_size), len(normalized_queries))
        similarity = normalized_queries[start:stop] @ normalized_candidates.T
        predicted_words.extend(candidate_words[np.argmax(similarity, axis=1)])
    return selected_words.tolist(), [str(word) for word in predicted_words], selected


def fixed_vocabulary_retrieval_metrics(
    query_embeddings,
    target_embeddings,
    target_words,
    vocabulary,
    top_ks=(1, 10),
    vocabulary_name="fixed",
):
    """只返回固定词表检索的汇总指标。"""
    summary, _, _ = fixed_vocabulary_retrieval(
        query_embeddings,
        target_embeddings,
        target_words,
        vocabulary,
        top_ks=top_ks,
        vocabulary_name=vocabulary_name,
    )
    return summary


def paired_cluster_bootstrap(
    clean_ranks,
    control_ranks,
    text_ids,
    cluster_ids,
    top_ks=(1, 5, 10),
    repetitions=2000,
    seed=42,
):
    """按物理显示行重采样，估计配对宏平均 Recall 差值区间。"""
    clean_ranks = np.asarray(clean_ranks)
    control_ranks = np.asarray(control_ranks)
    text_ids = np.asarray(text_ids)
    cluster_ids = np.asarray(cluster_ids)
    clusters = np.unique(cluster_ids)
    cluster_rows = {key: np.flatnonzero(cluster_ids == key) for key in clusters}
    random = np.random.default_rng(seed)
    bootstrap_differences = {int(k): np.empty(repetitions) for k in top_ks}

    for repetition in range(repetitions):
        sampled_clusters = random.choice(clusters, size=len(clusters), replace=True)
        sampled_rows = np.concatenate([cluster_rows[key] for key in sampled_clusters])
        sampled_ids = text_ids[sampled_rows]
        _, inverse = np.unique(sampled_ids, return_inverse=True)
        target_counts = np.bincount(inverse)
        for k in top_ks:
            clean_hits = clean_ranks[sampled_rows] <= int(k)
            control_hits = control_ranks[sampled_rows] <= int(k)
            clean_macro = np.mean(
                np.bincount(inverse, weights=clean_hits) / target_counts
            )
            control_macro = np.mean(
                np.bincount(inverse, weights=control_hits) / target_counts
            )
            bootstrap_differences[int(k)][repetition] = clean_macro - control_macro

    result = {
        "cluster_count": int(len(clusters)),
        "repetitions": int(repetitions),
        "seed": int(seed),
        "recall_differences": {},
    }
    for k in top_ks:
        clean_hits = clean_ranks <= int(k)
        control_hits = control_ranks <= int(k)
        differences = bootstrap_differences[int(k)]
        result["recall_differences"][f"macro_recall_at_{k}"] = {
            "estimate": mean_by_target(clean_hits, text_ids)
            - mean_by_target(control_hits, text_ids),
            "ci_95_low": float(np.quantile(differences, 0.025)),
            "ci_95_high": float(np.quantile(differences, 0.975)),
            "bootstrap_probability_above_zero": float(np.mean(differences > 0)),
        }
    return result
