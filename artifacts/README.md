# Local derived input artifacts

`artifacts/` 保存后续实验依赖、不能随实验结果清理而丢失的派生输入。

当前 `alignments/chineseeeg2_littleprince/{f1,m1}/` 保存实际朗读对齐资产。
二进制工作簿、CSV 和本地派生文件默认不进入 Git；可追溯的路径、大小与
SHA-256 记录在 `experiments/manifests/artifacts.json`。

这里不是模型输出目录，也不自动读取 `.env` 或下载外部资源。
