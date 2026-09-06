"""多数据集、多任务统一运行入口。"""

import argparse
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TASK_ROOT = PROJECT_ROOT / "tasks"


def discover_tasks():
    """发现带独立 train.py 的任务数据集目录。"""
    entries = {}
    for entry in TASK_ROOT.glob("*/*/train.py"):
        task_name = entry.parent.parent.name
        dataset_name = entry.parent.name
        entries[(task_name, dataset_name)] = entry
    return entries


def load_task_entry(entry_path):
    """加载任务目录自己的训练入口。"""
    module_name = f"task_{entry_path.parent.parent.name}_{entry_path.parent.name}"
    spec = spec_from_file_location(module_name, entry_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载任务入口：{entry_path}")
    module = module_from_spec(spec)
    sys.path.insert(0, str(entry_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def print_tasks(entries):
    """打印当前可运行的任务与数据集。"""
    print("可用任务")
    for (task_name, dataset_name), train_path in sorted(entries.items()):
        actions = ["prepare", "train"]
        if (train_path.parent / "evaluate.py").exists():
            actions.append("evaluate")
        print(f"  {task_name} / {dataset_name} ({', '.join(actions)})")


def parse_args(argv=None):
    """解析统一入口参数，其余参数原样交给任务 train.py。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="列出可用任务")
    parser.add_argument("task", nargs="?", help="任务类型，例如 sequence_decoding")
    parser.add_argument("dataset", nargs="?", help="数据集任务名称，例如 ChineseEEG_SR")
    parser.add_argument(
        "action",
        nargs="?",
        choices=("prepare", "train", "evaluate"),
        help="运行阶段",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="传给任务入口的参数")
    return parser.parse_args(argv), parser


def main(argv=None):
    """定位任务入口并执行数据准备或训练。"""
    args, parser = parse_args(argv)
    entries = discover_tasks()
    if args.list:
        print_tasks(entries)
        return
    if not args.task or not args.dataset or not args.action:
        parser.error("请提供 task、dataset 和 action，或使用 --list。")

    key = (args.task, args.dataset)
    if key not in entries:
        available = "、".join(f"{task}/{dataset}" for task, dataset in sorted(entries))
        parser.error(f"未知任务：{args.task}/{args.dataset}；可用项：{available or '无'}")
    entry_path = entries[key]
    if args.action == "evaluate":
        entry_path = entry_path.parent / "evaluate.py"
        if not entry_path.exists():
            parser.error(f"{args.task}/{args.dataset} 没有 evaluate.py。")
    task_module = load_task_entry(entry_path)
    forwarded = list(args.arguments)
    if args.action == "prepare":
        forwarded.insert(0, "--prepare")
    task_module.main(forwarded)


if __name__ == "__main__":
    main()
