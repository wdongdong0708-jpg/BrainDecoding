# Paper workspace rules

本文件约束 `paper/` 目录中的所有协作工作。

## Author ownership

论文正文由作者本人撰写。

除非用户明确要求，否则 AI 不得：

- 自动补写正文；
- 自动改写作者论点；
- 自动填写 Introduction、Results、Discussion 或 Methods；
- 自动生成摘要、结论、Figure caption 或实验结论。

## Source of truth

- `experiments/` 是科学实验协议和 frozen manifest 的唯一真源。
- `outputs/` 是单次实验运行结果的唯一真源。
- `reports/` 是跨实验导出和汇总数据的唯一真源。
- `paper/` 只负责学术叙事、论文写作和最终呈现。

不得把实验 manifest、运行 JSON、derived data 或 vocabulary manifest 复制到 `paper/`，也不得在此建立第二套结果真源。论文图表应消费 `reports/exports/` 中的正式导出。

## Quantitative claims

论文中的实验数字必须来自 canonical outputs 或 `reports/exports/`。

不得：

- 根据聊天记录或记忆填写数字；
- 在 plotting 脚本中手写结果数字；
- 为完成图表伪造 mock、random 或 placeholder 数字；
- 在来源、版本或计算合同不明确时补齐定量结论。

## Scientific language

- 区分 neural evidence 与 structural prior。
- 区分 exploratory 与 confirmatory。
- correct-minus-control 不称为 `pure neural information`。
- 不把声学可辨识性直接称为抽象语义解码。
- 不夸大结果。
- 不使用无依据的 breakthrough、revolutionary、unprecedented 等措辞。

## Figure principles

- 一张 Figure 尽量只回答一个主要问题。
- 正文 Figure panel 数量保持克制，完整诊断细节放入 Supplement。
- 优先输出 PDF/SVG 矢量图；PNG 仅用于预览或特定投稿要求。
- 不使用 3D 图、装饰性渐变或无语义的彩虹色。
- 相同条件在全文中保持一致的视觉语义。
- 数据精度优先于装饰。
- 图例、坐标轴和边框尽量减少视觉噪声。
- 在实际可行时保证灰度打印仍可辨识。

## Writing style

- concise
- precise
- restrained
- evidence-first
- avoid hype
- one paragraph, one clear function
- figure and text should complement rather than duplicate each other

以上规则只约束写作方式，不授权 AI 自动生成正文。
