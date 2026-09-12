# Canonical data pipeline

## 生命周期

```text
BRAINDATA_ROOT  第三方原始数据
artifacts/      人工或半人工形成的长期输入
derived/        可由 raw + artifacts 重建的数据产品
outputs/        可由 derived + configs 重建的单次模型结果
reports/        跨实验表格、图和导出
```

`experiments/build_derived_data.py` 是唯一 canonical 数据构建入口。缺少 derived
产品时明确报错，不读取 task-local cache。事件先构建，随后信号、文本和 manifest
验证；各数据集既有科学 builder、时间规则、划分和上下文定义保持不变。

## 当前 active 目标

| 数据集 | active scope | 计划保留的主要实验 |
|---|---|---|
| ChineseEEG2 Little Prince | `sub01-08` | `main_word`, `main_context` |
| SMN4Lang | `sub01-06` | `main_word`, `main_context` |
| LibriBrain100 | `sub0` | clean pipeline 后的主要 word/context baseline |

ChineseEEG2 的 subset scaling、SMN4Lang `sub01` development 和 LibriBrain100 的
历史消融不再属于机械重跑清单。

## 磁盘字段迁移

核心英文字段只在 loader 内存兼容层恢复，不进入 canonical CSV：

| 旧字段 | canonical 字段 | 主要消费者 | 状态 |
|---|---|---|---|
| `event_id` | `事件编号` | Dataset、评价、审计 | 磁盘已迁移；内存兼容 |
| `subject_id` | `受试者` | 受试者筛选、模型索引 | 磁盘已迁移；内存兼容 |
| `recording_id` | `记录编号` | 信号定位、审计控制 | 磁盘已迁移；内存兼容 |
| `word` | `词` | 可读事件内容 | 磁盘已迁移；内存兼容 |
| `normalized_word` | `标准词` | 文本向量、词表、评价 | 磁盘已迁移；内存兼容 |
| `sentence_uid` | `上下文编号` | sampler、context model | 磁盘已迁移；内存兼容 |
| `split` | `数据划分` | train/val/test 过滤 | 磁盘已迁移；内存兼容 |
| `is_trainable` | `是否可训练` | 事件资格过滤 | 磁盘已迁移；内存兼容 |
| `exclusion_reason` | `排除原因` | 事件审计 | 磁盘已迁移；内存兼容 |
| `aligned_start_seconds` / `onset_seconds` | `开始时间` | 信号窗口、排序 | 磁盘已迁移；内存兼容 |
| `aligned_stop_seconds` | `结束时间` | ChineseEEG2 时间审计 | 磁盘已迁移；内存兼容 |

其余被持久化的技术或数据集元数据也使用中文列名；对应英文名称由三个正式
数据模块的 `restore_event_table_columns()` 在内存中恢复。builder 中间列若不在
明确映射中会使 canonical 写入失败，避免无审计字段悄悄进入磁盘。

## 清理门槛

八人 ChineseEEG2、六人 SMN4Lang 和 LibriBrain100 的 events/signals/text 均通过
manifest 验证并完成主要 baseline 复核以后，才考虑删除：

- ChineseEEG2 subset-specific 事件表、重复 EEG/text cache 和旧 subset outputs；
- SMN4Lang task-local sub-01 cache，并可将其历史结果归入 `archive/smn4lang/sub01_legacy/`；
- LibriBrain100 task-local cache 与已由 clean rerun 替代的旧结果；
- 不再具有论文用途的 scaling、development、historical diagnostic 与 legacy alias 配置。

原始 EEG/MEG、alignment artifacts 和尚未完成复核的历史结果不在当前删除范围。

## 完整性检查

三个数据集使用同一个只读入口：

```powershell
python experiments/build_derived_data.py --dataset <dataset> --check
```

检查同时覆盖组件 manifest、自摘要、产品 SHA、事件到 recording 的一一覆盖、
可训练窗口边界、通道数和文本词序/shape/dtype。检查不会重建产品、生成 checkpoint
或执行模型评价。SMN4Lang 信号使用一次顺序读取 FIF 的 I/O 路径；滤波、降采样、
缩放、通道顺序和 float32 合同仍与 reference 路径相同。
