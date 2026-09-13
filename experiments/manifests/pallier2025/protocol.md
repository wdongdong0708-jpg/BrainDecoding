# Pallier2025 event protocol

本协议冻结 `pallier2025` 阶段 2A 的事件合同。原始数据为
`LittlePrince_MEG_French_Listen_Pallier2025`（OpenNeuro `ds007523`），正式
scope 为实际存在的 `sub-01` 至 `sub-10`。

主论文使用完整 run 作为材料与划分边界。`run_split.json` 使用固定 salt，
对 `run-01` 至 `run-09` 的标识计算 SHA-256 并按摘要升序排列：前 7 个为
train、第 8 个为 val、第 9 个为 test。该规则只读取 run 标识，不读取词、
词频、覆盖率、脑信号或模型结果。原 d'Ascoli sequence-level hash split 仅
保留为未来 reproduction 候选，不用于本阶段主 split。

`sub-09/run-03` 的原始 BIDS 文件不修改。不可变 QC artifact 证明其开头
annotation 与 stimulus/trigger/其他受试者不一致，但没有可恢复的神经时间；
因此事件表保留 BIDS 原行和时间，整条 recording 明确标为不可训练。

## 阶段 2B：连续 MEG 信号合同

参考实现冻结为 d'Ascoli commit
`e1262ee36aa2fbaa6965e5bfb893f6aaee0dd693` 的
`Meg._get_preprocessed_data()`。每条 recording 独立执行：

1. `mne.io.read_raw_fif(..., allow_maxshield=True)`；
2. 按冻结顺序选择 306 个 MEG 通道（102 magnetometers、204 planar
   gradiometers），不因 bad 标记删除通道；
3. MNE 0.1--40 Hz filter；
4. MNE 1000 Hz 到 50 Hz resample；
5. 在完整 recording 上按通道拟合 sklearn `RobustScaler`；
6. 转为 `float32` 后原子写入 canonical continuous signal。

该阶段不使用 notch、SSS/MaxFilter、ICA、artifact removal、baseline 或
clamp。Baseline（窗口前 0.5 秒）与 `[-5, 5]` clamp 属于后续 1 秒窗口提取
合同，不属于 continuous signal materialization。

BIDS onset 保持原 recording 时钟。canonical 样本索引按
`round((onset - first_samp / source_sfreq) * 50)` 计算，不假设
`first_samp == 0`。`sub-09/run-03` 的连续 MEG 可以物化，但其事件训练状态
继续按阶段 2A QC artifact 排除。

`run-06` raw FIF 允许仅为确定性预处理而读取；未加载模型、未生成 test
prediction、未计算或查看 test metric。

## 阶段 2C：词级分析窗口合同

Pallier2025 的 continuous MEG preprocessing follows the audited d'Ascoli
reference implementation。BrainDecoding 在此连续信号上有意采用 1.0 秒
post-onset neural input，同时保留固定 3.0 秒 event eligibility window，以维持
跨数据集词级解码协议。该分析窗口是 BrainDecoding 的 1 秒适配，不应描述为
“完全复现 d'Ascoli 的 3 秒模型输入”。

机器合同版本为 `word_1s_support_3s_baseline_0p5_v1`：

- Window：onset + 0.0 s 到 onset + 1.0 s，即 50 Hz 下 50 个样本；
- Eligibility：onset + 0.0 s 到 onset + 3.0 s；
- Baseline：1.0 秒神经窗口的前 0.5 秒，即前 25 个样本；
- Clamp：baseline correction 后限制到 `[-5, 5]`。

窗口提取顺序固定为：canonical continuous signal → 提取 onset 后 1.0 秒 →
减去前 0.5 秒逐通道均值 → clamp `[-5, 5]`。Baseline 与 clamp 都不属于
continuous signal materialization，既有 90 条 continuous signal 不因此重建。

候选词表只由 train run 的 material-level word occurrences 建立；同一故事词
位置不会因十名受试者重复计数。Story reference 独立覆盖 run-01 至 run-09，
仅定义目标故事语言分布，不参与候选词选择。Domain reference 在本阶段保持
`not_frozen`。
