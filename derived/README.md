# Derived data

`derived/` 保存能够从第三方原始数据与 `artifacts/` 完整重建的数据产品。
大型事件表、50 Hz 信号缓存、文本向量和运行时 provenance 默认不进入 Git。

固定生命周期：

```text
BRAINDATA_ROOT + artifacts
    -> events
    -> signals
    -> text
    -> manifest validation
```

统一入口：

```powershell
python experiments/build_derived_data.py --dataset chineseeeg2_littleprince --all
python experiments/build_derived_data.py --dataset smn4lang --all
python experiments/build_derived_data.py --dataset libribrain100 --all
```

canonical 模式不会回退到 `tasks/word_decoding/*/cache/`。旧配置仍可在清理前读取旧缓存。

只读完整性验证：

```powershell
python experiments/build_derived_data.py --dataset chineseeeg2_littleprince --check
python experiments/build_derived_data.py --dataset smn4lang --check
python experiments/build_derived_data.py --dataset libribrain100 --check
```

每个数据集根目录的 `manifest.json` 只在 events、signals、text 三个组件都通过
文件 SHA、记录覆盖、窗口边界和数据合同验证后标记为 `complete`。SMN4Lang 的
六人信号还要求严格达到 360/360 recording，单条信号和 sidecar 均以临时文件
完整写入后再原子替换。
