# ChineseEEG2 Little Prince 故事特异词汇审计 v0

This is an exploratory lexical audit. No model or test evaluation was performed.

本审计衡量相对于普通语言分布异常常见的词，即 story-specific lexical information；它不表示 semantic importance、剧情因果或人物重要度。

## 1. 数据集与材料计数

- dataset: `ChineseEEG2 Little Prince`
- language: `zh`
- frequency_unit: `voice_material_word_occurrence`
- deduplication_key: `['voice_version', 'source_word_id']`
- train story: `20405` tokens / `2180` types
- full story: `28123` tokens / `2640` types
- full story counts verified exactly equal to frozen `story_reference.json`.

## 2. Background source 与 coverage

- status: `blocked`
- background path: `not provided`
- source label: `missing_metadata`
- reason: `background_reference_not_provided`
- coverage: `not computed`
- tokenization_mismatch_risk: `not assessable without background`

仓库、`artifacts/`、本地 reference 配置及已知本地词频文件检索未找到可用的中文或法语 background。按照审计合同，本数据集不计算 story-specific score。

## 3. Candidate v0 formula

- `p_story(w) = story_count(w) / total_story_tokens`
- `p_bg_alpha(w) = (background_count(w) + alpha) / (total_background + alpha * |U|)`，其中 `U` 是 background 与 story 词型并集
- `raw_log_lift_bits = log2(p_story / p_bg_alpha)`
- `signed_kl_contribution_bits = p_story * raw_log_lift_bits`
- `positive_story_specific_score_bits = p_story * max(raw_log_lift_bits, 0)`

最后一项经过正值截断，不称为 KL contribution。

## 4. 未计算项目

smoothing stability、threshold Top50 排名/overlap、story-specific Top50、background OOV 高分词、q concentration、signed KL 与 positive score mass 均未计算。train-only threshold eligible type count 和 surface token flags 已完成，见相应报告。

## 5. Standard frequency Top50

`的`、`我`、`了`、`他`、`你`、`说`、`是`、`小王子`、`就`、`在`、`这`、`有`、`人`、`对`、`也`、`不`、`什么`、`很`、`着`、`都`、`会`、`一个`、`花儿`、`他们`、`她`、`上`、`呢`、`星星`、`它`、`去`、`又`、`要`、`地`、`得`、`那`、`没`、`个`、`过`、`来`、`把`、`可`、`想`、`国王`、`让`、`像`、`狐狸`、`和`、`星球`、`没有`、`问`

## 6. 是否建议冻结

`no`。background_reference_not_provided；不得冻结候选词表。

## Safeguards

- canonical frequency candidate vocabulary：未修改
- standard story reference：未修改
- split / event table / derived signals or text：未修改
- checkpoint / model forward / test neural arrays：未读取
- test prediction / test metric：未生成、未运行、未检查
