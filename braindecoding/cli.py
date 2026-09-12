"""BrainDecoding 的稳定命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from braindecoding.catalog import (
    discover_experiment_configs,
    experiment_selector,
    load_experiment,
    resolve_selector,
)
from braindecoding.config import PROJECT_ROOT
from braindecoding.experiment import (
    initialize_run_directory,
    run_directory,
    update_run_status,
    warm_start_checkpoint,
)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _run_status(record: dict) -> dict:
    """只读判断一个 canonical run 当前处于什么状态。"""
    config = record["raw_config"]
    output = run_directory(config)
    manifest = _read_json(output / "run_manifest.json")
    dependency = warm_start_checkpoint(config)
    if manifest is not None:
        status = manifest.get("status")
        if status not in {"running", "failed", "completed"}:
            status = "failed"
    elif output.exists():
        status = "failed"
    elif dependency is not None and not dependency.is_file():
        status = "waiting_for_upstream"
    else:
        status = "not_started"
    return {
        "status": status,
        "run_path": output,
        "manifest": manifest,
        "dependency": dependency,
        "best_checkpoint": output / "best.pt",
        "validation": output / "evaluation" / "val.json",
        "audit": output / "audits" / "validation" / "summary.json",
    }


def _print_list() -> int:
    print("dataset                         subject scope  experiment    status")
    for record in discover_experiment_configs():
        identity = record["identity"]
        status = _run_status(record)["status"]
        print(
            f"{identity['dataset']:<31} "
            f"{identity['subject_scope']:<14} "
            f"{identity['experiment_id']:<13} {status}"
        )
    return 0


def _task_modules(dataset: str):
    """延迟导入数据集任务，list/status 不加载训练依赖。"""
    if dataset == "chineseeeg2_littleprince":
        from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import (
            evaluate,
            train,
        )

        return train, evaluate
    if dataset == "smn4lang":
        from braindecoding.tasks.word_decoding.smn4lang import evaluate, train

        return train, evaluate
    if dataset == "libribrain100":
        from braindecoding.tasks.word_decoding.libribrain100 import evaluate, train

        return train, evaluate
    raise ValueError(f"当前 CLI 不支持数据集：{dataset}")


def _load_task_config(record: dict) -> tuple[dict, object, object]:
    train, evaluate = _task_modules(record["identity"]["dataset"])
    loader = getattr(train, "载入配置", None) or train.load_config
    return loader(record["config_path"]), train, evaluate


def _derived_status(dataset: str) -> str:
    manifest = _read_json(PROJECT_ROOT / "derived" / dataset / "manifest.json")
    return manifest.get("status", "missing") if manifest else "missing"


def _show(selector: str, resolved: bool) -> int:
    record = resolve_selector(selector)
    config, _, _ = _load_task_config(record)
    if resolved:
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        return 0
    identity = record["identity"]
    dataset = config["dataset"]
    training = config["training"]
    model = config["model"]
    budget = (
        f"{training['max_updates']} updates"
        if training.get("max_updates") is not None
        else f"{training['epochs']} epochs"
    )
    dependency = warm_start_checkpoint(config)
    print(f"identity: {record['selector']}")
    print(f"subjects: {', '.join(map(str, dataset.get('subjects', ())))}")
    print(f"derived status: {_derived_status(identity['dataset'])}")
    print(
        "window: "
        f"{dataset.get('window_start_offset_seconds', 0.0)} + "
        f"{dataset.get('window_seconds')} seconds"
    )
    print(
        "model: "
        f"use_transformer={model.get('use_transformer')}, "
        f"context_mode={model.get('context_mode')}"
    )
    print(f"training budget: {budget}")
    print(f"seed: {identity['seed']}")
    print(f"output: {run_directory(config)}")
    print(f"dependency: {dependency if dependency is not None else 'none'}")
    return 0


def _preflight(selector: str) -> int:
    from braindecoding.preflight import build_preflight_report

    record = resolve_selector(selector)
    report = build_preflight_report()
    item = next(
        value for value in report["experiments"] if value["config"] == record["relative_config"]
    )
    print(json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if item["ready"] in {True, "waiting_for_upstream"} else 1


def _missing_upstream(record: dict, config: dict) -> int | None:
    checkpoint = warm_start_checkpoint(config)
    if checkpoint is None or checkpoint.is_file():
        return None
    source = dict(record["identity"])
    source["experiment_id"] = config["training"]["warm_start_from"]
    selector = experiment_selector(source)
    print("Missing upstream experiment:", file=sys.stderr)
    print(f"  {selector}", file=sys.stderr)
    print("\nRun:", file=sys.stderr)
    print(f"  brain-decoding run {selector}", file=sys.stderr)
    return 2


def _train_once(dataset: str, train, config: dict):
    if dataset == "chineseeeg2_littleprince":
        summary = train.执行训练(config, save=True, force_cache=False)
        train.打印训练摘要(summary)
        return summary
    summary = train.run_training(config, save=True, force_cache=False)
    train.print_training_summary(summary)
    return summary


def _evaluate_once(dataset: str, evaluate, config: dict):
    checkpoint = Path(config["training"]["output_dir"]) / "best.pt"
    if dataset == "chineseeeg2_littleprince":
        summary = evaluate.评估检查点(
            config, checkpoint, split="val", save=True, force_cache=False
        )
        evaluate.打印评估摘要(summary)
        return summary
    summary = evaluate.evaluate_checkpoint(
        config, checkpoint, split="val", save=True, force_cache=False
    )
    evaluate.print_evaluation_summary(summary)
    return summary


def _run(selector: str) -> int:
    record = resolve_selector(selector)
    config, train, evaluate = _load_task_config(record)
    missing = _missing_upstream(record, config)
    if missing is not None:
        return missing

    from braindecoding.preflight import build_preflight_report

    report = build_preflight_report()
    item = next(
        value for value in report["experiments"] if value["config"] == record["relative_config"]
    )
    if item["ready"] is not True:
        print(
            "Preflight failed: " + ", ".join(item["reasons"] or ["unknown"]),
            file=sys.stderr,
        )
        return 2

    output, _ = initialize_run_directory(
        config,
        ["brain-decoding", "run", selector],
    )
    try:
        _train_once(record["identity"]["dataset"], train, config)
        _evaluate_once(record["identity"]["dataset"], evaluate, config)
    except BaseException:
        update_run_status(output, "failed")
        raise
    update_run_status(output, "completed")
    print(f"completed: {output}")
    return 0


def _evaluate(selector: str) -> int:
    record = resolve_selector(selector)
    config, _, evaluate = _load_task_config(record)
    checkpoint = Path(config["training"]["output_dir"]) / "best.pt"
    if not checkpoint.is_file():
        print(f"Missing checkpoint: {checkpoint}", file=sys.stderr)
        return 2
    _evaluate_once(record["identity"]["dataset"], evaluate, config)
    return 0


def _print_status(selector: str | None) -> int:
    records = (
        [resolve_selector(selector)] if selector else discover_experiment_configs()
    )
    for record in records:
        state = _run_status(record)
        print(f"{record['selector']}: {state['status']}")
        if selector:
            print(f"  run path: {state['run_path']}")
            print(f"  best checkpoint: {'present' if state['best_checkpoint'].is_file() else 'missing'}")
            print(f"  validation: {'present' if state['validation'].is_file() else 'not_run'}")
            print(f"  audit: {'present' if state['audit'].is_file() else 'not_run'}")
            print("  test: locked_not_evaluated")
    return 0


def _data_check(dataset: str) -> int:
    from braindecoding.data.build import check_dataset

    print(json.dumps(check_dataset(dataset), ensure_ascii=False, indent=2))
    return 0


def _data_build(args) -> int:
    from braindecoding.data.build import build_dataset

    selected = args.events or args.signals or args.text or args.all
    result = build_dataset(
        args.dataset,
        events=args.events or not selected or args.all,
        signals=args.signals or not selected or args.all,
        text=args.text or not selected or args.all,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _audit(selector: str, mapping_only: bool, device: str) -> int:
    record = resolve_selector(selector)
    dataset = record["identity"]["dataset"]
    names = {
        "chineseeeg2_littleprince": "ChineseEEG2",
        "smn4lang": "SMN4Lang",
    }
    if dataset not in names:
        print(f"audit unavailable for {dataset}", file=sys.stderr)
        return 2
    model_condition = (
        "word" if record["identity"]["experiment_id"] == "main_word" else "neural_context"
    )
    from braindecoding.audit.runner import run_dataset

    result = run_dataset(
        names[dataset],
        model_conditions=(model_condition,),
        device_name=device,
        mapping_only=mapping_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-decoding")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="列出 active canonical 实验")

    data = commands.add_parser("data", help="构建或检查 canonical derived 数据")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_check = data_commands.add_parser("check", help="只读验证 derived 数据")
    data_check.add_argument("dataset")
    data_build = data_commands.add_parser("build", help="构建 derived 数据")
    data_build.add_argument("dataset")
    data_build.add_argument("--events", action="store_true")
    data_build.add_argument("--signals", action="store_true")
    data_build.add_argument("--text", action="store_true")
    data_build.add_argument("--all", action="store_true")
    data_build.add_argument("--force", action="store_true")

    show = commands.add_parser("show", help="显示实验摘要")
    show.add_argument("experiment")
    show.add_argument("--resolved", action="store_true")
    preflight = commands.add_parser("preflight", help="只读执行训练前检查")
    preflight.add_argument("experiment")
    run = commands.add_parser("run", help="训练并执行 validation 评价")
    run.add_argument("experiment")
    evaluate = commands.add_parser("evaluate", help="只执行 validation 评价")
    evaluate.add_argument("experiment")
    status = commands.add_parser("status", help="查看实验运行状态")
    status.add_argument("experiment", nargs="?")
    audit = commands.add_parser("audit", help="运行现有 validation control audit")
    audit.add_argument("experiment")
    audit.add_argument("--mapping-only", action="store_true")
    audit.add_argument("--device", default="auto")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return _print_list()
        if args.command == "data":
            return _data_check(args.dataset) if args.data_command == "check" else _data_build(args)
        if args.command == "show":
            return _show(args.experiment, args.resolved)
        if args.command == "preflight":
            return _preflight(args.experiment)
        if args.command == "run":
            return _run(args.experiment)
        if args.command == "evaluate":
            return _evaluate(args.experiment)
        if args.command == "status":
            return _print_status(args.experiment)
        if args.command == "audit":
            return _audit(args.experiment, args.mapping_only, args.device)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    raise AssertionError(f"未处理的命令：{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
