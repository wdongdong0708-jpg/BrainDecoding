# BrainDecoding

BrainDecoding 使用固定命令管理 canonical 数据和实验；日常使用不需要直接运行仓库内部的 Python 文件。

## 快速开始

先配置原始神经数据和本地文本模型的根目录，然后以 editable 模式安装：

```powershell
$env:BRAINDATA_ROOT = "D:/dataset"
$env:BRAINDECODING_MODEL_ROOT = "D:/code/dascoli-word-decoding/models"
pip install -e .
```

OVMI 是可选依赖，并固定到已审计提交：

```powershell
pip install -e ".[ovmi]"
```

常用命令：

```text
brain-decoding list
brain-decoding data check chineseeeg2_littleprince
brain-decoding show chineseeeg2_littleprince/sub01-08/main_word
brain-decoding preflight chineseeeg2_littleprince/sub01-08/main_word
brain-decoding run chineseeeg2_littleprince/sub01-08/main_word
brain-decoding evaluate chineseeeg2_littleprince/sub01-08/main_word
brain-decoding status
```

`python -m braindecoding ...` 与 `brain-decoding ...` 完全等价。`run` 会先执行 preflight，再训练并只评价 validation；`evaluate` 也固定为 validation，不提供 test 参数。

## 数据

```text
BRAINDATA_ROOT/  第三方原始数据，不属于仓库
artifacts/       人工或半人工生成、后续实验长期依赖的输入资产
derived/         可由 raw + artifacts 重新构建的数据产品
outputs/         canonical 模型运行结果
reports/         跨实验表格、图和导出报告
```

检查已有 derived 数据：

```text
brain-decoding data check chineseeeg2_littleprince
brain-decoding data check smn4lang
brain-decoding data check libribrain100
```

构建全部数据产品：

```text
brain-decoding data build smn4lang --all
```

也可以分别使用 `--events`、`--signals`、`--text`。只有明确审查来源合同变化后才使用 `--force`。

## 实验 selector

CLI 统一使用：

```text
dataset/subject_scope/experiment_id
```

当前 active 实验由 `configs/word_decoding/` 自动发现。运行目录固定为：

```text
outputs/{task}/{dataset}/{subject_scope}/{experiment_id}/seed-{seed:03d}/
```

上下文实验若缺少同 scope、同 seed 的 `main_word/best.pt`，CLI 会给出应先运行的 selector，不会自动启动依赖。

## 目录职责

- `braindecoding/`：安装后的 Python 包、CLI、数据、任务、训练与评价实现。
- `configs/`：canonical 科学配置。
- `experiments/`：冻结协议、manifest 和审计资产，不作为用户脚本入口。
- `scripts/maintenance/`：低频迁移或历史维护工具。
- `tests/`：科学行为、配置、数据合同与 CLI 回归测试。

正式运行默认要求 Git tracked 工作树干净。未跟踪或 `.gitignore` 管理的本地 `artifacts/`、`derived/` 和 `outputs/` 不会触发该保护。
