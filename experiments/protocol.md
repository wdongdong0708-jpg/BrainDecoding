# ChineseEEG2 / SMN4Lang 词级论文实验协议（审计草案）

状态：`core_decisions_machine_frozen_preconfirmatory`  
日期：2026-09-11  
范围：ChineseEEG2 PassiveListening 与 SMN4Lang；LibriBrain100 仅保留历史兼容；不含 LittlePrince MEG。

本文档记录已经冻结的核心协议和仍未冻结的控制参数；它不是已经完成的确认性
实验注册。本文档同时记录已经查看过的验证/测试信息，避免把探索性结果误称为
盲测结果。机器可检查资产位于 `experiments/manifests/`。

## 0. 已冻结的核心决定

- ChineseEEG2 `word` 使用 `model.use_transformer=false`；正式
  `neural_context` 使用 `bounded_semantic_v1`。ROW 仅作为历史/结构先验对照。
- SMN4Lang `word` 使用 `model.use_transformer=false`；正式
  `neural_context` 使用官方 script 句子及连续块上下文。
- 主结果候选词表规模固定为 20、50、100、150；200、300、500 仅为
  supplementary exploratory，不属于主结果必跑集合。
- 候选词只由 train 材料生成；评价 split 的支持情况不得改变词或 N。
- domain reference 尚未冻结；temporal shift 的具体偏移参数尚未冻结。

## 1. 数据和受试者范围

### ChineseEEG2

- 数据集元数据与本机预处理目录均包含 `sub-01` 至 `sub-08`。
- 论文主分析拟使用全部八名被动聆听受试者。
- 当前八人配置将 `f1` 分配给 `sub-01` 至 `sub-04`，将 `m1` 分配给
  `sub-05` 至 `sub-08`；现有八人事件缓存包含全部八名受试者。
- 第 14、27 章因 `sub-05` 对应记录截断而对八名受试者共同排除。

### SMN4Lang

- OpenNeuro `ds004078` 1.2.1 的 `participants.tsv` 列出 `sub-01` 至
  `sub-12`，这是论文拟使用的全体可用受试者。
- 本机当前只下载了 `sub-01` 的 60 个预处理 MEG recording；因此尚不能验证
  12 人事件数、通道一致性、信号缓存或真实训练。
- 当前默认配置明确为 `subjects: [sub-01]`。加载器按配置遍历受试者，并不会
  自动发现 `participants.tsv` 中的全部受试者。
- 当前本机目录为 `D:/dataset/SMN4Lang`，而公共配置在
  `BRAINDATA_ROOT=D:/dataset` 时指向 `D:/dataset/ds004078`；全受试者运行前
  必须先明确数据目录约定，但本轮不修改配置。

多受试者事件构建的主体逻辑是安全的：`recording_id/event_id/sentence_uid`
包含受试者或 recording，事件按 `subject_id/run/onset/word_index` 稳定排序；
材料编号和划分单元只使用 story，因此不会按受试者跨 split。训练词表对 run
去重，不会把同一故事因受试者重复而重复计数。

仍需消除的单受试者假设：单人 expected counts、单人 cache/output 名、默认
subjects、单人 warm-start checkpoint 形状，以及 evaluate 对 checkpoint 中
具体 subject ID/顺序缺少显式核验。12 人数据尚未下载也是当前硬阻塞。

## 2. 材料级 split

划分边界固定使用中文事件合同的 `划分单元`；同一划分单元跨所有受试者只能
属于一个 split。

### ChineseEEG2

| split | 配置分配章节 | 当前可训练章节 |
|---|---|---|
| train | 01, 04, 06, 07, 08, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27 | 前述章节去掉 27 |
| val | 03, 05, 25 | 03, 05, 25 |
| test | 02, 09, 14 | 02, 09（14 被共同排除） |

划分单元格式为 `ChineseEEG2|littleprince|chapter-XX`，不含受试者和 voice。
当前缓存验证为每个上下文最多对应一个 split 和一个 chapter。

### SMN4Lang

| split | 完整 story/run |
|---|---|
| train | 01–50 |
| val | 51–55 |
| test | 56–60 |

划分单元格式为 `SMN4Lang|story-XX`，不含受试者。当前单人缓存的 60 个 story
均有可训练事件，且没有跨 split。

## 3. Train-only 候选词表支持审计

候选词只按训练材料中的 `标准词` 频率生成；频次降序，频次相同按词典序升序。
本节查看 val/test 仅用于报告支持风险，绝不用于重新选词、改变顺序或决定 N。

ChineseEEG2 的词频单位是去重后的 `(voice_version, source_word_id)`，不会把四名
受试者重复计数。train/val/test 可训练材料词数分别为 20,405 / 3,533 / 2,120。

| N | train 截止频次 | train 覆盖 | val 支持/覆盖 | test 支持/覆盖 |
|---:|---:|---:|---:|---:|
| 20 | 132 | 31.82% | 20/20；31.93% | 20/20；29.81% |
| 50 | 60 | 44.70% | 47/50；43.59% | 46/50；40.61% |
| 100 | 32 | 54.92% | 88/100；52.19% | 82/100；50.42% |
| 150 | 22 | 61.25% | 122/150；57.88% | 108/150；56.93% |
| 200 | 16 | 65.71% | 151/200；61.65% | 131/200；61.75% |
| 300 | 10 | 71.79% | 204/300；66.37% | 155/300；65.33% |
| 500 | 6 | 78.94% | 261/500；71.78% | 199/500；70.71% |

ChineseEEG2 Top-50 的零样本词：val 为 `国王/她/星星`；test 为
`他们/国王/星星/狐狸`。

SMN4Lang 的候选频次单位是训练 story 中去重后的 `(run, word_index)` 非空词，
共 36,046 个；覆盖率分母使用具有完整窗口的 35,810 / 3,846 / 3,385 个
train/val/test 事件。以下统计来自当前 `sub-01` 缓存；12 人数据尚未物化。

| N | train 截止频次 | train 覆盖 | val 支持/覆盖 | test 支持/覆盖 |
|---:|---:|---:|---:|---:|
| 20 | 147 | 21.88% | 20/20；22.00% | 20/20；20.53% |
| 50 | 64 | 30.15% | 48/50；29.46% | 49/50；28.27% |
| 100 | 38 | 37.03% | 94/100；34.92% | 95/100；35.39% |
| 150 | 28 | 41.55% | 133/150；38.66% | 133/150；39.62% |
| 200 | 23 | 45.11% | 168/200；41.19% | 161/200；42.51% |
| 300 | 17 | 50.66% | 230/300；45.06% | 234/300；48.01% |
| 500 | 11 | 58.13% | 342/500；50.75% | 347/500；54.18% |

SMN4Lang Top-50 的零样本词：val 为 `教育/直播`；test 为 `直播`。

结论：两个数据集只有本次审计的 Top-20 在 val/test 都具有完整类别支持。
这不是选择 Top-20 的依据；N 仍需仅根据训练与验证阶段的预注册目标决定。

## 4. 候选词表冻结文件

每个数据集、每个预注册 N 分别生成一个 `vocabulary.json`：

```json
{
  "vocabulary": ["..."],
  "vocabulary_size": 50,
  "source_split": "train",
  "source_split_units": ["..."],
  "policy": "frequency_desc_word_asc",
  "normalization": "dataset_standard_word_v1",
  "word_counts": {"...": 123},
  "created_from_test": false
}
```

文件必须在确认性 test 前冻结并记录 SHA-256。val/test 支持不能改变其内容。

## 5. OVMI reference

### story reference

- ChineseEEG2：使用 f1 与 m1 两个完整 27 章刺激版本；按各自材料顺序统计
  `标准词`，每个 `(voice_version, source_word_id)` 只计一次，不按受试者重复。
  包含 train/val/test，也包含因神经记录原因排除的第 14、27 章。该定义回答
  “完整目标故事语言分布”，不参与候选词或模型选择。当前全文共 28,123 词次、
  2,640 词型。
- SMN4Lang：使用 story 01–60 的全部标准词，每个 `(run, word_index)` 只计一次，
  包含 train/val/test，不按受试者重复。当前标注共 43,327 非空词次、9,122 词型。

story reference 使用全部目标材料属于评价目标分布定义，不等于使用 test 挑词。
仍须把其来源、去重键、normalization、内容指纹和 split-independent 用途写入
provenance。

### domain reference

本轮不指定具体语料。正式文件必须满足：与实验故事独立、语言匹配、自然叙事
领域匹配、分词规则明确、normalization 与事件 `标准词` 一致、来源/许可/版本/
下载日期/处理代码可追溯。缺失的 `references/subtlex_ch.json` 不是正式方案。

## 6. full OVMI 的零样本词限制

固定版本的官方 OVMI full 实现会对选中的混淆矩阵逐行归一化，并要求每个选中
行具有正质量；因此每个候选词必须至少有一个真实评价样本。当前适配层只是提前
检测并返回明确的 unavailable 状态，不是它自行新增的数学限制。

如果 train-only 候选在 test 中没有样本：

1. 固定候选词表和 retrieval 结果保持不变；不得按 test 重选。
2. 该 N 的完整固定词表 full OVMI 报告为不可用，并列出零样本词。
3. 可以另报“observed-support conditional OVMI”，但它是 test-support 条件化的
   次级描述结果，必须换名称，不能替代主结果或跨 N 直接比较。
4. 平滑零行、scalar OVMI 或其他估计方法均属于新指标合同；只有预注册后才能用。

## 7. 模型条件

- `word`：`model.use_transformer: false`。每个词的脑窗口独立经过 brain encoder；
  sampler 分组不会进入预测计算。
- `neural_context`：`model.use_transformer: true` 且
  `model.context_mode: grouped`。每个词先独立脑编码，再按冻结的 `上下文编号`
  进入 grouped Transformer。
- `context_mode: singleton` 是带单元素 Transformer 的历史诊断，不等同于正式
  `word` 基线。
- 文本模型只提供目标 embedding；两个正式条件均不允许语言模型概率重排。

## 8. 上下文合同和结构风险

| 数据集/历史条件 | 上下文来源 | 组数 | 平均词数 | 中位数 | 最大值 | 跨划分单元 |
|---|---|---:|---:|---:|---:|---|
| ChineseEEG2 ROW | 实际朗读工作簿行；连续块上限 128 | 20,812 | 5.01 | 5 | 16 | 否 |
| ChineseEEG2 semantic_v1 | 跨行标点语义片段；目标 16、上限 32 词/15 秒 | 10,268 | 10.15 | 9 | 31 | 否 |
| SMN4Lang | 官方 script 非空句；连续块上限 128 | 1,827 | 23.56 | 21 | 100 | 否 |

ChineseEEG2 正式 `neural_context` 已冻结为 semantic_v1；ROW 仅保留为历史和
结构先验对照。两者都有组长、绝对位置、相对位置先验；现有 validation 结构
基线明显高于随机。SMN4Lang 的 script sentence 同样具有强位置/长度结构：现有
validation 审计中，组合结构基线宏 Top-10 约 27%–28%（全支持），在严格支持
子集上可达约 52%。这些都是探索性结果，不能作为确认性效应。

## 9. 统一审计矩阵

| 模型 | 条件 | ChineseEEG2 | SMN4Lang | 正式状态 |
|---|---|---|---|---|
| word | clean | 已有 CNN-only 路径 | 已有 CNN-only 路径 | 需冻结全受试者配置 |
| word | temporal_shift | 缺失 | 缺失 | 需要实现 |
| word | donor_swap | 有零散 CNN 特征诊断 | 有零散 CNN 特征诊断 | 需要统一实现 |
| neural_context | clean | 已有 | 已有 | 需冻结上下文合同/全受试者配置 |
| neural_context | temporal_shift | 缺失；现有 within-row 不是时间平移 | 同左 | 需要实现 |
| neural_context | donor_swap | 已有 validation 特征置换 | 已有 validation 特征置换 | 探索性；需统一查询支持 |
| neural_context | structure_only | 已有长度/绝对/相对位置逻辑回归 | 已有 | 探索性；需冻结拟合合同 |
| neural_context | donor_following | 已有 validation 审计 | 已有 validation 审计 | 探索性；需统一输出 |

统一输出要求：Top-1、Top-10、macro retrieval、median rank 和 MRR 已有；mean rank
当前公共汇总没有，需要实现；coverage、OVMI story、OVMI domain 只有 reference
冻结且 full 支持条件满足时才可报告。不得伪造不可用项。

## 10. 时间错位与 donor swap 待冻结定义

### temporal_shift

主建议：在同一受试者、同一 recording 内按 `记录内序号` 做确定性循环错位，
使用预注册的事件偏移；要求 donor 窗口与原窗口不重叠，并保持原评价目标。
固定偏移应在 train/val 上选定，不能根据 test 效果调整。若 recording 太短而无法
满足条件，该查询在所有比较条件中共同排除，不能保留正确输入混入对照。

### donor_swap

主对照 donor 来自同一受试者、同一 recording、相同窗口长度的另一个可训练词
事件；必须 `donor 标准词 != target 标准词` 且窗口不重叠。评价目标、上下文组、
组内位置和 subject index 保持原样。建议固定随机种子 `0..19` 报告均值与范围，
但范围不是置信区间；统计推断另按原始材料事件成簇 bootstrap。跨 recording donor
只作为次级诊断。最终应冻结共同可替换查询，禁止某些对照保留 clean 输入。

`donor_following` 报告 donor 词的 Top-1/Top-10、rank improvement，并与原目标
性能下降同时报告。

## 11. 测试数据历史和确认性纪律

- ChineseEEG2：至少 `sub-01` 至 `sub-04` 的 test EEG 已用于 150 词曲线和混淆
  机制审计；当前 test 必须标记 `已用于探索`，不能再称盲测。八人完整 test 是否
  全部运行过不能据现有证据断言，但材料标签和部分神经数据已经暴露。
- SMN4Lang：sub-01 至 sub-06 的全部 raw FIF（包括 test split recording）已经为
  canonical deterministic signal materialization 读取。该访问只执行冻结的确定性
  预处理，没有加载模型、生成 test prediction 或计算/查看 test model metric；因此
  状态是“raw test neural data accessed for deterministic preprocessing only”，不能再称
  `unopened`，也不能称已经进行 test model evaluation。test 词标签此前还用于支持审计。
- checkpoint 选择当前使用 validation 固定 50 词 macro Top-10；正式协议若改变
  选择指标，必须在任何确认性 test 前完成并冻结。

只有以下项目全部冻结并写入带哈希的 manifest 后，才允许一次性运行确认性 test：

- 数据预处理、事件资格和 split
- 全受试者列表及 subject index 顺序
- candidate vocabulary、N 和 vocabulary JSON
- text embedding
- word/neural_context 模型结构及上下文定义
- 训练预算、随机种子和初始化
- validation checkpoint 选择规则
- temporal_shift/donor_swap/structure-only 审计协议
- story/domain reference 及文件哈希
- 查询支持规则和统计汇总方式

test 结果产生后，不允许反向修改上述项目。ChineseEEG2 当前 split 的后续结果只能
作为复现/探索性结果；若需要真正确认性证据，应另行预注册未使用数据或外部复现。

## 12. 确认性运行前仍未冻结的事项

1. temporal shift 的确切偏移和最小不重叠间隔；偏移只能用 train/val 冻结。
2. 两个数据集 domain reference 的独立语料来源。
3. full OVMI 不可用时是否预注册独立命名的支持条件化次级指标。
4. donor swap 是否增加跨 recording 次级条件；主条件固定为同受试者、同记录，
   随机种子固定为 0 至 19。
5. SMN4Lang 12 人数据完整性、expected counts、cache 名和 checkpoint subject 合同。
