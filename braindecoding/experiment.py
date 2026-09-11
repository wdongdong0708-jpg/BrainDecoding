"""实验身份、运行目录与可复现清单工具。"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

from braindecoding.config import PROJECT_ROOT


EXPERIMENT_CATEGORIES = (
    "main",
    "ablation",
    "scaling",
    "development",
    "historical_diagnostic",
)
_IDENTITY_KEYS = ("task", "dataset", "subject_scope", "id")
_RUN_STATUSES = ("running", "failed", "completed")
_IDENTITY_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def experiment_identity(config):
    """验证并返回稳定的实验五元组。"""
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("canonical 配置缺少 experiment 身份。")
    missing = [key for key in _IDENTITY_KEYS if not experiment.get(key)]
    if missing:
        raise ValueError(f"experiment 身份缺少字段：{', '.join(missing)}")
    invalid = [
        key
        for key in _IDENTITY_KEYS
        if not _IDENTITY_COMPONENT.fullmatch(str(experiment[key]))
    ]
    if invalid:
        raise ValueError(
            "experiment 身份字段只能使用小写字母、数字、下划线和连字符："
            + ", ".join(invalid)
        )
    category = experiment.get("category")
    if category not in EXPERIMENT_CATEGORIES:
        raise ValueError(f"experiment.category 不受支持：{category}")
    training = config.get("training")
    if not isinstance(training, dict) or "seed" not in training:
        raise ValueError("canonical 配置必须显式提供 training.seed。")
    raw_seed = training["seed"]
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ValueError("training.seed 必须为非负整数。")
    seed = raw_seed
    if seed < 0:
        raise ValueError("training.seed 必须为非负整数。")
    return {
        "task": str(experiment["task"]),
        "dataset": str(experiment["dataset"]),
        "subject_scope": str(experiment["subject_scope"]),
        "experiment_id": str(experiment["id"]),
        "seed": seed,
    }


def run_directory(config, output_root=None):
    """仅由实验五元组推导单次运行目录。"""
    identity = experiment_identity(config)
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    return (
        root
        / identity["task"]
        / identity["dataset"]
        / identity["subject_scope"]
        / identity["experiment_id"]
        / f"seed-{identity['seed']:03d}"
    )


def warm_start_checkpoint(config, output_root=None):
    """解析同任务、数据集、受试者范围和 seed 下的 warm-start 检查点。"""
    warm_start_from = config.get("training", {}).get("warm_start_from")
    if not warm_start_from:
        return None
    source_config = copy.deepcopy(config)
    source_config["experiment"]["id"] = str(warm_start_from)
    return run_directory(source_config, output_root=output_root) / "best.pt"


def evaluation_output_path(config, split, *, smoke=False):
    """返回评价文件位置；legacy 配置继续使用原文件名。"""
    output_dir = Path(config["training"]["output_dir"])
    suffix = "_smoke" if smoke else ""
    if "experiment" in config:
        return output_dir / "evaluation" / f"{split}{suffix}.json"
    return output_dir / f"evaluation_{split}{suffix}.json"


def resolve_experiment_config(config, output_root=None):
    """向 canonical 配置注入派生路径；旧配置保持原样。"""
    resolved = copy.deepcopy(config)
    if "experiment" not in resolved:
        return resolved
    output_dir = str(run_directory(resolved, output_root=output_root))
    resolved.setdefault("training", {})["output_dir"] = output_dir
    if "closed_set_diagnostic" in resolved:
        resolved["closed_set_diagnostic"]["output_dir"] = output_dir
    checkpoint = warm_start_checkpoint(resolved, output_root=output_root)
    if checkpoint is not None:
        resolved["training"]["pretrained_brain_encoder_checkpoint"] = str(
            checkpoint
        )
    return resolved


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolved_config_sha256(config):
    """计算与字典键顺序无关的 resolved config SHA-256。"""
    payload = json.dumps(
        _json_ready(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    """计算文件 SHA-256；文件不存在时返回 ``None``。"""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _path_record(path):
    path = Path(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def build_run_manifest(
    config,
    command,
    status="running",
    protocol_manifests=(),
    vocabulary_manifests=(),
):
    """构造不主动读取测试数据的单次运行清单。"""
    if status not in _RUN_STATUSES:
        raise ValueError(f"运行状态不受支持：{status}")
    identity = experiment_identity(config)
    event_table = config.get("cache", {}).get("event_table")
    if event_table is None:
        event_table = config.get("training", {}).get("event_table")
    return {
        "experiment_identity": identity,
        "resolved_config_sha256": resolved_config_sha256(config),
        "git_commit": _git_commit(),
        "command": command if isinstance(command, str) else list(command),
        "seed": identity["seed"],
        "event_table": _path_record(event_table) if event_table else None,
        "protocol_manifests": [
            _path_record(path) for path in protocol_manifests
        ],
        "vocabulary_manifests": [
            _path_record(path) for path in vocabulary_manifests
        ],
        "status": status,
    }


def initialize_run_directory(
    config,
    command,
    *,
    resume=False,
    output_root=None,
    protocol_manifests=(),
    vocabulary_manifests=(),
):
    """建立标准运行资产，并拒绝含糊覆盖已有目录。"""
    resolved = resolve_experiment_config(config, output_root=output_root)
    output_dir = run_directory(resolved, output_root=output_root)
    manifest_path = output_dir / "run_manifest.json"
    config_hash = resolved_config_sha256(resolved)

    if output_dir.exists():
        if not manifest_path.is_file():
            raise FileExistsError(
                f"运行目录已存在但缺少 run_manifest.json，拒绝覆盖：{output_dir}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("resolved_config_sha256") != config_hash:
            raise FileExistsError(f"运行目录中的配置 SHA 不同，拒绝覆盖：{output_dir}")
        if existing.get("status") == "completed":
            raise FileExistsError(f"运行已经完成，拒绝再次覆盖：{output_dir}")
        if not resume:
            raise FileExistsError(
                f"已有 {existing.get('status')} 运行；只有显式 resume 才可继续：{output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)

    manifest = build_run_manifest(
        resolved,
        command,
        status="running",
        protocol_manifests=protocol_manifests,
        vocabulary_manifests=vocabulary_manifests,
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "evaluation").mkdir(exist_ok=True)
    (output_dir / "audits" / "validation").mkdir(parents=True, exist_ok=True)
    return output_dir, manifest


def update_run_status(output_dir, status):
    """更新已有运行清单状态，不改写其他 provenance。"""
    if status not in _RUN_STATUSES:
        raise ValueError(f"运行状态不受支持：{status}")
    manifest_path = Path(output_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到运行清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
