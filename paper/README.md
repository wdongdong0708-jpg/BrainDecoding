# Paper workspace

`paper/` 是最终论文写作、图形设计和学术呈现工作区。它不保存实验协议、单次运行结果或跨实验原始汇总数据。

数据职责边界：

- `experiments/`：科学协议与 frozen manifests。
- `outputs/`：单次 canonical run 结果。
- `reports/`：跨实验汇总、论文数据导出和图表数据。
- `paper/`：LaTeX 稿件、绘图代码、最终 Figure/Table 和作者笔记。

主要位置：

- `manuscript/`：LaTeX 主文件和章节文件。
- `plotting/`：只从 `reports/exports/` 消费正式数据的绘图脚本。
- `figures/`：最终论文 Figure 文件。
- `tables/`：最终论文 Table 文件。
- `references/references.bib`：作者核对并维护的参考文献。
- `notes/`：作者的故事线、决定、实验记录和开放问题。

如果本机已经安装 LaTeX，可以执行：

```bash
cd paper/manuscript
latexmk -pdf main.tex
```

本仓库不会安装 LaTeX，也暂不绑定任何期刊模板。

## Regenerate paper figures

```bash
python -m braindecoding.paper_exports
python paper/plotting/make_all.py --split val
```

未来切换到正式 test 数据：

```bash
python -m braindecoding.paper_exports
python paper/plotting/make_all.py --split test
```

Paper plotting scripts never read raw experiment JSON directly; they only read
the CSV files in `reports/exports/`.
