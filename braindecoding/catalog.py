"""发现并解析当前 active canonical 实验。"""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path

import yaml

from braindecoding.config import PROJECT_ROOT, deep_merge, load_yaml_with_extends
from braindecoding.experiment import experiment_identity, resolve_experiment_config


def experiment_selector(identity: dict) -> str:
    """把实验身份转换为用户使用的稳定 selector。"""
    return "/".join(
        (
            identity["dataset"],
            identity["subject_scope"],
            identity["experiment_id"],
        )
    )


def _load_raw_config(path: Path, ancestors=()) -> dict:
    """只为发现实验读取继承关系，不展开机器环境变量。"""
    path = path.resolve()
    if path in ancestors:
        raise ValueError(f"配置继承出现循环：{path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return deep_merge(_load_raw_config(parent_path, (*ancestors, path)), raw)


def discover_experiment_configs(config_root=None) -> list[dict]:
    """从 canonical 配置树发现 active main 实验，不硬编码实验清单。"""
    root = Path(config_root or PROJECT_ROOT / "configs" / "word_decoding")
    records = []
    for path in sorted(root.rglob("*.yaml")):
        raw = _load_raw_config(path)
        experiment = raw.get("experiment")
        if not isinstance(experiment, dict) or experiment.get("category") != "main":
            continue
        identity = experiment_identity(raw)
        records.append(
            {
                "selector": experiment_selector(identity),
                "identity": identity,
                "config_path": path.resolve(),
                "relative_config": path.resolve()
                .relative_to(PROJECT_ROOT.resolve())
                .as_posix(),
                "raw_config": raw,
            }
        )
    selectors = [record["selector"] for record in records]
    if len(selectors) != len(set(selectors)):
        raise ValueError("canonical 配置中存在重复实验 selector。")
    return records


def resolve_selector(selector: str, config_root=None) -> dict:
    """解析用户 selector；不存在时给出最接近的有效候选。"""
    records = discover_experiment_configs(config_root)
    by_selector = {record["selector"]: record for record in records}
    if selector in by_selector:
        return by_selector[selector]
    candidates = sorted(by_selector)
    nearby = get_close_matches(selector, candidates, n=3, cutoff=0.25)
    hint = nearby or candidates
    rendered = "\n  ".join(hint) if hint else "<无>"
    raise ValueError(f"未知实验：{selector}\n可用或相近候选：\n  {rendered}")


def load_experiment(selector: str, config_root=None, output_root=None) -> tuple[dict, dict]:
    """载入 selector 对应的完整 canonical 配置。"""
    record = resolve_selector(selector, config_root)
    config = load_yaml_with_extends(record["config_path"])
    return record, resolve_experiment_config(config, output_root=output_root)
