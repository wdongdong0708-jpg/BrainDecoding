# BrainDecoding 历史实验重跑与命名迁移计划

状态：`canonical_identity_frozen_legacy_outputs_untouched`  
日期：2026-09-11

本文件只定义实验身份、配置入口、目标运行目录和依赖关系，不认可或重算任何
历史指标。历史 `outputs/` 保持原地不动；canonical 配置在本阶段继续通过
`extends` 引用精确的 legacy 科学配置，以避免人工复制参数造成漂移。因此旧配置
已停止作为新运行入口，但在下一次“自包含配置物化 + 全量等价性测试”完成前不得
移动到 `configs/legacy/`。

## 1. 固定身份与目录规则

每次运行的唯一身份为：

```text
(task, dataset, subject_scope, experiment_id, seed)
```

输出目录只由该五元组产生：

```text
outputs/{task}/{dataset}/{subject_scope}/{experiment_id}/seed-{seed:03d}/
```

`experiment.category` 只允许 `main`、`ablation`、`scaling`、
`development`、`historical_diagnostic`。固定模型名、embedding 维数、训练步数、
seed、受试者和非消融窗口不得进入 experiment ID。

## 2. ChineseEEG2 LittlePrince

| legacy config | legacy output（本机） | canonical config | experiment ID | canonical output | category | rerun | dependency |
|---|---|---|---|---|---|---|---|
| `ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_cnn_only.yaml` | `sub-01_to_sub-08/actual_reading_1s/cnn_only`（checkpoint + val） | `word_decoding/chineseeeg2_littleprince/sub01-08/main_word.yaml` | `main_word` | `word_decoding/chineseeeg2_littleprince/sub01-08/main_word/seed-000` | main | rerun | — |
| `ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_semantic_cnn_warm_start.yaml` | `sub-01_to_sub-08/actual_reading_1s/semantic_context_cnn_warm_start`（checkpoint + val + audit） | `word_decoding/chineseeeg2_littleprince/sub01-08/main_context.yaml` | `main_context` | `word_decoding/chineseeeg2_littleprince/sub01-08/main_context/seed-000` | main | rerun | `main_word` |
| `ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s_cnn_warm_start.yaml` | `sub-01_to_sub-08/actual_reading_1s/cnn_warm_start`（checkpoint + val + audit） | `word_decoding/chineseeeg2_littleprince/sub01-08/ablation_context_row.yaml` | `ablation_context_row` | `word_decoding/chineseeeg2_littleprince/sub01-08/ablation_context_row/seed-000` | ablation | rerun | `main_word` |
| `ChineseEEG2_LittlePrince_sub01_sub08_actual_reading_1s.yaml` | `sub-01_to_sub-08/actual_reading_1s/base`（未找到运行文件） | `word_decoding/chineseeeg2_littleprince/sub01-08/ablation_context_scratch.yaml` | `ablation_context_scratch` | `word_decoding/chineseeeg2_littleprince/sub01-08/ablation_context_scratch/seed-000` | ablation | rerun | — |
| `ChineseEEG2_LittlePrince_sub01_actual_reading_1s_cnn_only.yaml` | `sub-01/actual_reading_1s/cnn_only`（checkpoint） | `word_decoding/chineseeeg2_littleprince/sub01/scaling_word.yaml` | `scaling_word` | `word_decoding/chineseeeg2_littleprince/sub01/scaling_word/seed-000` | scaling | rerun | — |
| `ChineseEEG2_LittlePrince_sub01_actual_reading_1s_cnn_warm_start.yaml` | `sub-01/actual_reading_1s/cnn_warm_start`（checkpoint + val + audit） | `word_decoding/chineseeeg2_littleprince/sub01/scaling_context_row.yaml` | `scaling_context_row` | `word_decoding/chineseeeg2_littleprince/sub01/scaling_context_row/seed-000` | scaling | rerun | `scaling_word` |
| `ChineseEEG2_LittlePrince_sub01_actual_reading_1s.yaml` | `sub-01/actual_reading_1s/base`（未找到运行文件） | `word_decoding/chineseeeg2_littleprince/sub01/ablation_context_scratch.yaml` | `ablation_context_scratch` | `word_decoding/chineseeeg2_littleprince/sub01/ablation_context_scratch/seed-000` | ablation | rerun | — |
| `ChineseEEG2_LittlePrince_sub01_sub02_actual_reading_1s_cnn_only.yaml` | `sub-01_sub-02/actual_reading_1s/cnn_only`（checkpoint） | `word_decoding/chineseeeg2_littleprince/sub01-02/scaling_word.yaml` | `scaling_word` | `word_decoding/chineseeeg2_littleprince/sub01-02/scaling_word/seed-000` | scaling | rerun | — |
| `ChineseEEG2_LittlePrince_sub01_sub02_actual_reading_1s_cnn_warm_start.yaml` | `sub-01_sub-02/actual_reading_1s/cnn_warm_start`（checkpoint + val + audit） | `word_decoding/chineseeeg2_littleprince/sub01-02/scaling_context_row.yaml` | `scaling_context_row` | `word_decoding/chineseeeg2_littleprince/sub01-02/scaling_context_row/seed-000` | scaling | rerun | `scaling_word` |
| `ChineseEEG2_LittlePrince_sub01_sub02_actual_reading_1s.yaml` | `sub-01_sub-02/actual_reading_1s/base`（未找到运行文件） | `word_decoding/chineseeeg2_littleprince/sub01-02/ablation_context_scratch.yaml` | `ablation_context_scratch` | `word_decoding/chineseeeg2_littleprince/sub01-02/ablation_context_scratch/seed-000` | ablation | rerun | — |
| `ChineseEEG2_LittlePrince_sub01_sub04_actual_reading_1s_cnn_only.yaml` | `sub-01_to_sub-04/actual_reading_1s/cnn_only`（checkpoint） | `word_decoding/chineseeeg2_littleprince/sub01-04/scaling_word.yaml` | `scaling_word` | `word_decoding/chineseeeg2_littleprince/sub01-04/scaling_word/seed-000` | scaling | rerun | — |
| `ChineseEEG2_LittlePrince_sub01_sub04_actual_reading_1s_cnn_warm_start.yaml` | `sub-01_to_sub-04/actual_reading_1s/cnn_warm_start`（checkpoint + val/test + audit） | `word_decoding/chineseeeg2_littleprince/sub01-04/scaling_context_row.yaml` | `scaling_context_row` | `word_decoding/chineseeeg2_littleprince/sub01-04/scaling_context_row/seed-000` | scaling | rerun | `scaling_word` |
| `ChineseEEG2_LittlePrince_sub01_sub04_actual_reading_1s.yaml` | `sub-01_to_sub-04/actual_reading_1s/base`（checkpoint + val + audit） | `word_decoding/chineseeeg2_littleprince/sub01-04/ablation_context_scratch.yaml` | `ablation_context_scratch` | `word_decoding/chineseeeg2_littleprince/sub01-04/ablation_context_scratch/seed-000` | ablation | rerun | — |
| `ChineseEEG2_LittlePrince_sub05_sub08_actual_reading_1s_cnn_only.yaml` | `sub-05_to_sub-08/actual_reading_1s/cnn_only`（checkpoint + val） | `word_decoding/chineseeeg2_littleprince/sub05-08/scaling_word.yaml` | `scaling_word` | `word_decoding/chineseeeg2_littleprince/sub05-08/scaling_word/seed-000` | scaling | rerun | — |
| `ChineseEEG2_LittlePrince_sub05_sub08_actual_reading_1s_cnn_warm_start.yaml` | `sub-05_to_sub-08/actual_reading_1s/cnn_warm_start`（checkpoint + val + audit） | `word_decoding/chineseeeg2_littleprince/sub05-08/scaling_context_row.yaml` | `scaling_context_row` | `word_decoding/chineseeeg2_littleprince/sub05-08/scaling_context_row/seed-000` | scaling | rerun | `scaling_word` |
| `ChineseEEG2_LittlePrince_sub05_sub08_actual_reading_1s.yaml` | `sub-05_to_sub-08/actual_reading_1s/base`（未找到运行文件） | `word_decoding/chineseeeg2_littleprince/sub05-08/ablation_context_scratch.yaml` | `ablation_context_scratch` | `word_decoding/chineseeeg2_littleprince/sub05-08/ablation_context_scratch/seed-000` | ablation | rerun | — |

`ChineseEEG2_LittlePrince.yaml` 与
`ChineseEEG2_LittlePrince_sub01_actual_reading_1s.yaml` 的递归展开结果完全相同；
它们只对应一个 `sub01/ablation_context_scratch` 可运行 identity，不创建第二个
canonical 别名。

## 3. SMN4Lang（当前仅 sub-01 development）

| legacy config | legacy output（本机） | canonical config | experiment ID | canonical output | category | rerun | dependency |
|---|---|---|---|---|---|---|---|
| `SMN4Lang_1s_conv_only.yaml` | `word_decoding_all_words_mengzi_1s_conv_only_6400updates`（checkpoint + val） | `word_decoding/smn4lang/sub01/dev_word.yaml` | `dev_word` | `word_decoding/smn4lang/sub01/dev_word/seed-000` | development | rerun development | — |
| `SMN4Lang_1s.yaml` | `word_decoding_all_words_mengzi_1s_sentence_transformer_6400updates`（checkpoint + val） | `word_decoding/smn4lang/sub01/dev_context.yaml` | `dev_context` | `word_decoding/smn4lang/sub01/dev_context/seed-000` | development | rerun development | — |
| `SMN4Lang_1s_cnn_warm_start.yaml` | `word_decoding_all_words_mengzi_1s_cnn_warm_start_freeze960_6400updates`（checkpoint + val + audit） | `word_decoding/smn4lang/sub01/ablation_context_warm_start.yaml` | `ablation_context_warm_start` | `word_decoding/smn4lang/sub01/ablation_context_warm_start/seed-000` | ablation | rerun development | `dev_word` |
| `SMN4Lang_1s_single_word_transformer.yaml` | `word_decoding_all_words_mengzi_1s_single_word_transformer_6400updates`（checkpoint + val） | `word_decoding/smn4lang/sub01/ablation_context_singleton.yaml` | `ablation_context_singleton` | `word_decoding/smn4lang/sub01/ablation_context_singleton/seed-000` | ablation | rerun development | — |
| `SMN4Lang_conv_only.yaml` | `word_decoding_all_words_sentence_groups_conv_only_6400updates`（checkpoint + val） | `word_decoding/smn4lang/sub01/ablation_window_3s_word.yaml` | `ablation_window_3s_word` | `word_decoding/smn4lang/sub01/ablation_window_3s_word/seed-000` | ablation | rerun development | — |
| `SMN4Lang.yaml` | `word_decoding_all_words_sentence_groups_6400updates`（checkpoint + val） | `word_decoding/smn4lang/sub01/ablation_window_3s_context.yaml` | `ablation_window_3s_context` | `word_decoding/smn4lang/sub01/ablation_window_3s_context/seed-000` | ablation | rerun development | — |
| `SMN4Lang_3s_single_word_transformer.yaml` | `word_decoding_all_words_mengzi_3s_single_word_transformer_6400updates`（checkpoint + val） | `word_decoding/smn4lang/sub01/ablation_window_3s_singleton.yaml` | `ablation_window_3s_singleton` | `word_decoding/smn4lang/sub01/ablation_window_3s_singleton/seed-000` | ablation | rerun development | — |
| `SMN4Lang_gpt2_conv_only.yaml` | `word_decoding_all_words_gpt2_layer24_1024_conv_only_6400updates`（checkpoint + val） | `word_decoding/smn4lang/sub01/ablation_text_gpt2_word.yaml` | `ablation_text_gpt2_word` | `word_decoding/smn4lang/sub01/ablation_text_gpt2_word/seed-000` | ablation | rerun development | — |
| `SMN4Lang_gpt2.yaml` | `word_decoding_all_words_gpt2_layer24_1024_6400updates`（checkpoint，无 val 文件） | `word_decoding/smn4lang/sub01/ablation_text_gpt2_context.yaml` | `ablation_text_gpt2_context` | `word_decoding/smn4lang/sub01/ablation_text_gpt2_context/seed-000` | ablation | rerun development | — |

不创建 `sub01-12/main_word` 或 `sub01-12/main_context`：当前多人数据与相应科学
配置尚未存在，伪配置不可运行也不可作为正式身份。

## 4. LibriBrain100 历史兼容回归

| legacy config | legacy output（本机） | canonical config | experiment ID | canonical output | category | rerun | dependency |
|---|---|---|---|---|---|---|---|
| `LibriBrain100_1s_conv_only.yaml` | `word_decoding_1s_conv_only`（checkpoint + val） | `word_decoding/libribrain100/sub0/historical_word_1s.yaml` | `historical_word_1s` | `word_decoding/libribrain100/sub0/historical_word_1s/seed-000` | historical_diagnostic | compatibility rerun | — |
| `LibriBrain100_1s.yaml` | `word_decoding_1s_sentence_transformer`（checkpoint + val + audit） | `word_decoding/libribrain100/sub0/historical_grouped_1s.yaml` | `historical_grouped_1s` | `word_decoding/libribrain100/sub0/historical_grouped_1s/seed-000` | historical_diagnostic | compatibility rerun | — |
| `LibriBrain100.yaml` | `word_decoding`（checkpoint + val） | `word_decoding/libribrain100/sub0/historical_grouped_3s.yaml` | `historical_grouped_3s` | `word_decoding/libribrain100/sub0/historical_grouped_3s/seed-000` | historical_diagnostic | compatibility rerun | — |
| `LibriBrain100_1s_cnn_warm_start.yaml` | `word_decoding_1s_cnn_warm_start`（checkpoint + val + audit） | `word_decoding/libribrain100/sub0/historical_grouped_warm_start_1s.yaml` | `historical_grouped_warm_start_1s` | `word_decoding/libribrain100/sub0/historical_grouped_warm_start_1s/seed-000` | historical_diagnostic | compatibility rerun | `historical_word_1s` |
| `LibriBrain100_1s_single_word_transformer.yaml` | `word_decoding_1s_single_word_transformer`（checkpoint + val） | `word_decoding/libribrain100/sub0/historical_singleton_1s.yaml` | `historical_singleton_1s` | `word_decoding/libribrain100/sub0/historical_singleton_1s/seed-000` | historical_diagnostic | compatibility rerun | — |
| `LibriBrain100_3s_single_word_transformer.yaml` | `word_decoding_3s_single_word_transformer`（checkpoint + val） | `word_decoding/libribrain100/sub0/historical_singleton_3s.yaml` | `historical_singleton_3s` | `word_decoding/libribrain100/sub0/historical_singleton_3s/seed-000` | historical_diagnostic | compatibility rerun | — |

## 5. ChineseEEG1 SR 历史诊断

| legacy config | legacy output（本机） | canonical config | experiment ID | canonical output | category | rerun | dependency |
|---|---|---|---|---|---|---|---|
| `ChineseEEG_SR.yaml` 的 `training` 条件 | `LittlePrince_fourier_subject_row_retrieval`（checkpoint） | `sequence_decoding/chineseeeg1_sr/sub04-10_sub13-14/historical_row_retrieval_fourier_subject.yaml` | `historical_row_retrieval_fourier_subject` | `sequence_decoding/chineseeeg1_sr/sub04-10_sub13-14/historical_row_retrieval_fourier_subject/seed-042` | historical_diagnostic | compatibility rerun | — |
| `ChineseEEG_SR.yaml` 的 `closed_set_diagnostic` 条件 | `LittlePrince_closed_set_loso`（diagnostic summary） | `sequence_decoding/chineseeeg1_sr/sub04-10_sub13-14/historical_closed_set_loso.yaml` | `historical_closed_set_loso` | `sequence_decoding/chineseeeg1_sr/sub04-10_sub13-14/historical_closed_set_loso/seed-042` | historical_diagnostic | compatibility rerun | — |

两个 canonical 文件通过 `run_section` 明确选择原单一 YAML 中的科学条件；该字段
不改变任何模型或数据参数。

## 6. 仅归档、不可重跑的 orphan

| legacy output | status | reason |
|---|---|---|
| `outputs/ChineseEEG1_SR/LittlePrince_row_retrieval` | `archive_only` | `exact_config_unavailable` |
| `outputs/SMN4Lang/word_decoding` | `archive_only` | `exact_config_unavailable` |
| `outputs/SMN4Lang/word_decoding_sentence_groups_6400updates` | `archive_only` | `exact_config_unavailable` |

这些目录不对应任何可运行 canonical 配置，不得根据目录名反推参数。

## 7. 不属于 canonical 训练 run 的本地输出

- `outputs/女声一小王子时间戳/` 与 `outputs/男声一小王子时间戳/` 是带人工/工具链
  处理痕迹的对齐输入资产，不是可任意删除的模型输出。后续应迁到
  `artifacts/chineseeeg2_littleprince/alignment/{f1,m1}/`，本轮不移动。
- `outputs/audits/`、各历史 run 下的 `audit_*` 与词表曲线目录是已有派生审计结果，
  后续归入对应 canonical run 的 `audits/validation/` 或明确的历史归档，本轮不移动。
- `outputs/pytest_*`、`outputs/01a08422-c069-7810-977a-92b259884b04/` 是本地测试/
  临时产物，不构成实验 identity。
- `outputs/observed_vocabulary_confusion_validation/` 是独立验证诊断，不构成训练 run。

## 8. 标准 run 文件布局

```text
seed-000/
  resolved_config.yaml
  run_manifest.json
  best.pt
  last.pt
  training_summary.json
  evaluation/
    val.json
    val_n20.json
    val_n50.json
    val_n100.json
    val_n150.json
  audits/
    validation/
```

只有获得后续明确授权时才写 `evaluation/test*.json`。多个冻结词表 N 是同一
checkpoint 的评价条件，不产生四个训练目录。

`run_manifest.json` 记录五元组、resolved config SHA-256、Git commit、启动命令、
seed、事件表路径/摘要、适用的 protocol/vocabulary manifest 摘要和运行状态。
四个现有训练 CLI 在 canonical 配置且允许保存时，会在训练前初始化这两份文件，
并在正常结束或异常退出后把状态更新为 `completed` 或 `failed`；legacy 配置、
`--prepare`、纯评价与 `--no-save` 保持原行为。
目标目录不存在时可创建；目录缺少 manifest、配置 SHA 不同或状态为 completed
时拒绝覆盖；running/failed 只能显式 resume。禁止使用 `_new`、`_v2`、`_final`
绕过冲突。

## 9. 重跑顺序

1. 先运行所有无依赖的 word/scratch/singleton/window/text 条件。
2. 完成并核验 ChineseEEG2 各 scope 的 `main_word`/`scaling_word` 后，再运行相同
   scope 与 seed 的 warm-start context。
3. 完成 SMN4Lang `dev_word` 后，再运行 `ablation_context_warm_start`。
4. 完成 LibriBrain100 `historical_word_1s` 后，再运行
   `historical_grouped_warm_start_1s`。
5. 每个 checkpoint 只先执行 validation；test 仍受冻结协议控制。

## 10. 科学行为保护

canonical/legacy 等价检查忽略的字段仅限 `experiment`、派生 `output_dir`、
`warm_start_from` 及其派生 checkpoint 路径。受试者与顺序、split、事件表、信号与
资格窗口、上下文、文本表示、模型、优化器、scheduler、训练预算、冻结时长、seed、
词表和评价协议必须逐值相同。最危险的漂移点是把 semantic 与 ROW 对调、把 1 秒
输入和 3 秒资格窗口混为一谈、改变 sub01-04/sub05-08 的人员或 voice、让 warm-start
跨 scope/seed、以及把历史 test 结果写入新的 validation 身份。
