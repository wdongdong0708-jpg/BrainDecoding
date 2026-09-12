"""生成一次性清理清单；仅在显式 ``--apply`` 时按冻结清单执行。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends
from braindecoding.experiment import (
    resolve_experiment_config,
    scientific_config_sha256,
)


ACTIVE_CONFIGS = {
    "chineseeeg2_main_word": (
        "configs/word_decoding/chineseeeg2_littleprince/"
        "sub01-08/main_word.yaml"
    ),
    "chineseeeg2_main_context": (
        "configs/word_decoding/chineseeeg2_littleprince/"
        "sub01-08/main_context.yaml"
    ),
    "smn4lang_main_word": (
        "configs/word_decoding/smn4lang/sub01-06/main_word.yaml"
    ),
    "smn4lang_main_context": (
        "configs/word_decoding/smn4lang/sub01-06/main_context.yaml"
    ),
    "libribrain100_main_word": (
        "configs/word_decoding/libribrain100/sub0/main_word.yaml"
    ),
    "libribrain100_main_context": (
        "configs/word_decoding/libribrain100/sub0/main_context.yaml"
    ),
}

SCIENTIFIC_CONFIG_SHA256 = {
    "chineseeeg2_main_word": "3117f60305b0aa6de8dd930e896ab9fa3ab26829acbab28a3cdb89e9f863573e",
    "chineseeeg2_main_context": "3951b8a5d1936265da3d506ada1e74aeb5e2a34ae4466a4cb6fbdf70150d6387",
    "smn4lang_main_word": "40fba01a86059b0b1299498b0bbff50aca2b8b0080995936604afbf84103ca1a",
    "smn4lang_main_context": "91cb836820b6d1e49077396f19eea013f703c20e2ee0b68d62f5501b3677a9d2",
    "libribrain100_main_word": "af6c317dcc0b8e697504d69f887765762bf20394f7f1f7786470dfc6de951617",
    "libribrain100_main_context": "bd27d21f8a7ea54c380483a019596223eee5144625a927322319bdd0e6d62385",
}

DERIVED_MANIFESTS = {
    "chineseeeg2_littleprince": "derived/chineseeeg2_littleprince/manifest.json",
    "smn4lang": "derived/smn4lang/manifest.json",
    "libribrain100": "derived/libribrain100/manifest.json",
}

CACHE_ROOTS = {
    "tasks/word_decoding/ChineseEEG2_LittlePrince/cache": (
        "derived/chineseeeg2_littleprince",
        "derived/chineseeeg2_littleprince/manifest.json",
    ),
    "tasks/word_decoding/SMN4Lang/cache": (
        "derived/smn4lang",
        "derived/smn4lang/manifest.json",
    ),
    "tasks/word_decoding/LibriBrain100/cache": (
        "derived/libribrain100",
        "derived/libribrain100/manifest.json",
    ),
}

DELETE_OUTPUT_ROOTS = (
    "outputs/ChineseEEG2_LittlePrince/sub-01",
    "outputs/ChineseEEG2_LittlePrince/sub-01_sub-02",
    "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-04",
    "outputs/ChineseEEG2_LittlePrince/sub-05_to_sub-08",
    "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-08/actual_reading_1s/cnn_warm_start",
    "outputs/LibriBrain100/word_decoding",
    "outputs/LibriBrain100/word_decoding_1s_cnn_warm_start",
    "outputs/LibriBrain100/word_decoding_1s_single_word_transformer",
    "outputs/LibriBrain100/word_decoding_3s_single_word_transformer",
    "outputs/女声一小王子时间戳",
    "outputs/男声一小王子时间戳",
)

ARCHIVE_ROOTS = {
    "outputs/SMN4Lang": "archive/smn4lang/sub01_legacy/SMN4Lang",
    "outputs/audits": "reports/legacy/validation_controls",
    "outputs/observed_vocabulary_confusion_validation": (
        "reports/legacy/observed_vocabulary_confusion_validation"
    ),
    "outputs/01a08422-c069-7810-977a-92b259884b04": (
        "reports/legacy/exports/01a08422-c069-7810-977a-92b259884b04"
    ),
    "outputs/本对话数据表格汇总.md": "reports/legacy/本对话数据表格汇总.md",
    "outputs/一秒上下文检查点跨数据集审计.md": (
        "reports/legacy/一秒上下文检查点跨数据集审计.md"
    ),
    "outputs/word_decoding_context_window_ablation.json": (
        "reports/legacy/word_decoding_context_window_ablation.json"
    ),
}

KEEP_OUTPUT_ROOTS = (
    "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-08/actual_reading_1s/cnn_only",
    "outputs/ChineseEEG2_LittlePrince/sub-01_to_sub-08/actual_reading_1s/semantic_context_cnn_warm_start",
    "outputs/LibriBrain100/word_decoding_1s_conv_only",
    "outputs/LibriBrain100/word_decoding_1s_sentence_transformer",
    # sequence decoding 明确留待独立清理。
    "outputs/ChineseEEG1_SR",
)


def _json_bytes(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _manifest_sha256(value) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _relative(path: Path) -> str:
    return path.absolute().relative_to(PROJECT_ROOT.absolute()).as_posix()


def _files(path: Path):
    if path.is_file():
        return (path,)
    if not path.is_dir():
        return ()
    result = []
    for directory, names, files in os.walk(path, followlinks=False):
        # Windows junction 可能指向仓库外；清单和删除都不得遍历其目标。
        names[:] = [
            name
            for name in names
            if not (
                getattr(
                    (Path(directory) / name).stat(follow_symlinks=False),
                    "st_file_attributes",
                    0,
                )
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ]
        result.extend(Path(directory) / name for name in files)
    return tuple(sorted(result))


def _entry(
    path: Path,
    *,
    category: str,
    action: str,
    canonical_replacement=None,
    replacement_manifest=None,
    destination=None,
    legacy_manifest_covered=None,
):
    if action not in {"delete", "archive", "keep"}:
        raise ValueError(f"未知清理动作：{action}")
    result = {
        "path": _relative(path),
        "size_bytes": path.stat().st_size,
        "category": category,
        "canonical_replacement": canonical_replacement,
        "replacement_manifest": replacement_manifest,
        "active_reference_count": 0,
        "action": action,
        "reason": {
            "delete": "canonical replacement complete or reproducible temporary output",
            "archive": "historical evidence retained outside active outputs",
            "keep": "temporary legacy numeric reference or separately scoped sequence result",
        }[action],
    }
    if destination is not None:
        result["destination"] = destination
    if legacy_manifest_covered is not None:
        result["legacy_manifest_covered"] = legacy_manifest_covered
    return result


def _active_config_audit() -> dict:
    os.environ.setdefault("BRAINDATA_ROOT", "D:/dataset")
    os.environ.setdefault(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )
    result = {}
    for name, relative in ACTIVE_CONFIGS.items():
        path = PROJECT_ROOT / relative
        raw = path.read_text(encoding="utf-8")
        first = raw.splitlines()[0]
        if first != "extends: ../base.yaml":
            raise ValueError(f"active config 仍依赖非 canonical base：{relative}")
        if "output_dir:" in raw:
            raise ValueError(f"active config 手写 output_dir：{relative}")
        config = resolve_experiment_config(load_yaml_with_extends(path))
        cache_values = tuple(str(value).replace("\\", "/") for value in config["cache"].values())
        if not all(value.startswith("derived/") for value in cache_values):
            raise ValueError(f"active config 仍引用非 derived 数据：{relative}")
        digest = scientific_config_sha256(config)
        if digest != SCIENTIFIC_CONFIG_SHA256[name]:
            raise ValueError(
                f"active config 科学合同漂移：{relative}；{digest} != "
                f"{SCIENTIFIC_CONFIG_SHA256[name]}"
            )
        result[name] = {
            "config": relative,
            "scientific_config_sha256_before": SCIENTIFIC_CONFIG_SHA256[name],
            "scientific_config_sha256_after": digest,
            "cache_paths": list(cache_values),
        }
    return result


def _derived_audit() -> dict:
    result = {}
    for dataset, relative in DERIVED_MANIFESTS.items():
        path = PROJECT_ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"canonical derived 尚未 complete：{relative}")
        result[dataset] = {
            "manifest": relative,
            "status": payload["status"],
            "manifest_sha256": payload.get("manifest_sha256"),
        }
    return result


def _legacy_outputs() -> tuple[dict, set[str]]:
    path = PROJECT_ROOT / "experiments/legacy_results_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_count") != 32 or len(payload.get("runs", ())) != 32:
        raise ValueError("legacy result manifest 未覆盖冻结的 32 个科学运行。")
    return payload, {str(run["legacy_output"]).replace("\\", "/") for run in payload["runs"]}


def _covered(path: Path, legacy_outputs: set[str]) -> bool:
    relative = _relative(path)
    return any(relative == root or relative.startswith(root + "/") for root in legacy_outputs)


def _obsolete_configs() -> tuple[Path, ...]:
    root = PROJECT_ROOT / "configs"
    root_legacy = []
    for pattern in (
        "ChineseEEG2_LittlePrince*.yaml",
        "SMN4Lang*.yaml",
        "LibriBrain100*.yaml",
    ):
        root_legacy.extend(root.glob(pattern))
    active = {str((PROJECT_ROOT / path).resolve()) for path in ACTIVE_CONFIGS.values()}
    bases = {
        str((PROJECT_ROOT / "configs/word_decoding" / dataset / "base.yaml").resolve())
        for dataset in ("chineseeeg2_littleprince", "smn4lang", "libribrain100")
    }
    inactive = [
        path
        for path in (PROJECT_ROOT / "configs/word_decoding").rglob("*.yaml")
        if str(path.resolve()) not in active | bases
    ]
    return tuple(sorted(set(root_legacy + inactive)))


def build_cleanup_manifest() -> dict:
    """完成全部前置验证并返回逐文件 dry-run 清单。"""
    active = _active_config_audit()
    derived = _derived_audit()
    legacy, legacy_outputs = _legacy_outputs()
    entries = []

    for relative, (replacement, replacement_manifest) in CACHE_ROOTS.items():
        for path in _files(PROJECT_ROOT / relative):
            entries.append(
                _entry(
                    path,
                    category="task_local_cache",
                    action="delete",
                    canonical_replacement=replacement,
                    replacement_manifest=replacement_manifest,
                )
            )

    for relative in DELETE_OUTPUT_ROOTS:
        root = PROJECT_ROOT / relative
        for path in _files(root):
            entries.append(
                _entry(
                    path,
                    category=(
                        "alignment_temporary"
                        if "时间戳" in relative
                        else "inactive_legacy_run"
                    ),
                    action="delete",
                    canonical_replacement=(
                        "artifacts/alignments/chineseeeg2_littleprince"
                        if "时间戳" in relative
                        else "experiments/legacy_results_manifest.json"
                    ),
                    replacement_manifest=(
                        "experiments/manifests/artifacts.json"
                        if "时间戳" in relative
                        else "experiments/legacy_results_manifest.json"
                    ),
                    legacy_manifest_covered=_covered(path, legacy_outputs),
                )
            )

    for pytest_root in sorted((PROJECT_ROOT / "outputs").glob("pytest_*")):
        for path in _files(pytest_root):
            entries.append(
                _entry(path, category="pytest_temporary", action="delete")
            )

    for source, destination in ARCHIVE_ROOTS.items():
        root = PROJECT_ROOT / source
        for path in _files(root):
            suffix = path.resolve().relative_to(root.resolve()).as_posix() if root.is_dir() else path.name
            target = destination if root.is_file() else f"{destination}/{suffix}"
            entries.append(
                _entry(
                    path,
                    category="legacy_report_or_run",
                    action="archive",
                    destination=target,
                    legacy_manifest_covered=_covered(path, legacy_outputs),
                )
            )

    for relative in KEEP_OUTPUT_ROOTS:
        root = PROJECT_ROOT / relative
        for path in _files(root):
            entries.append(
                _entry(
                    path,
                    category="temporary_numeric_reference",
                    action="keep",
                    legacy_manifest_covered=_covered(path, legacy_outputs),
                )
            )

    for path in _obsolete_configs():
        entries.append(
            _entry(
                path,
                category="obsolete_config",
                action="delete",
                canonical_replacement="six active canonical configs",
            )
        )

    entries.sort(key=lambda item: (item["action"], item["path"]))
    counts = {
        action: sum(item["action"] == action for item in entries)
        for action in ("delete", "archive", "keep")
    }
    bytes_by_action = {
        action: sum(
            item["size_bytes"] for item in entries if item["action"] == action
        )
        for action in ("delete", "archive", "keep")
    }
    manifest = {
        "schema_version": 1,
        "status": "dry_run_validated_before_cleanup",
        "generator": "experiments/generate_cleanup_manifest.py",
        "active_configs": active,
        "derived_replacements": derived,
        "legacy_result_manifest": {
            "path": "experiments/legacy_results_manifest.json",
            "run_count": legacy["run_count"],
            "manifest_sha256": legacy.get("manifest_sha256"),
        },
        "counts": counts,
        "bytes_by_action": bytes_by_action,
        "entries": entries,
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


def _summary_value(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    best = payload.get("best_validation") or {}
    return {
        "status": payload.get("status"),
        "best_score": payload.get("best_score"),
        "best_epoch": payload.get("best_epoch"),
        "best_update": payload.get("best_update") or payload.get("optimizer_updates"),
        "selection_metric": payload.get("selection_metric"),
        "validation": {
            key: best.get(key)
            for key in (
                "query_count",
                "median_rank",
                "mean_reciprocal_rank",
                "micro_recall_at_1",
                "micro_recall_at_10",
                "macro_recall_at_1",
                "macro_recall_at_10",
            )
            if key in best
        },
    }


def write_legacy_summaries(legacy: dict) -> None:
    """只摘取轻量指标；checkpoint 仍仅由 immutable manifest 记录 SHA。"""
    output = PROJECT_ROOT / "reports" / "legacy"
    output.mkdir(parents=True, exist_ok=True)
    prefixes = {
        "chineseeeg2": "chineseeeg2_",
        "smn4lang": "smn4lang_",
        "libribrain100": "libribrain100_",
    }
    for dataset, prefix in prefixes.items():
        runs = []
        for run in legacy["runs"]:
            if not str(run["legacy_run_id"]).startswith(prefix):
                continue
            summaries = []
            for record in run.get("summary_files", ()):
                path = PROJECT_ROOT / record["relative_path"]
                summaries.append(
                    {
                        "path": record["relative_path"],
                        "sha256": record["sha256"],
                        "selected_values": _summary_value(path),
                    }
                )
            runs.append(
                {
                    "legacy_run_id": run["legacy_run_id"],
                    "legacy_output": run["legacy_output"],
                    "canonical_identity": run.get("canonical_identity"),
                    "status": run["status"],
                    "summaries": summaries,
                    "checkpoint_files": run.get("checkpoint_files", ()),
                    "evaluation_files": run.get("evaluation_files", ()),
                    "audit_files": run.get("audit_files", ()),
                }
            )
        payload = {
            "schema_version": 1,
            "dataset": dataset,
            "source": "experiments/legacy_results_manifest.json",
            "source_manifest_sha256": legacy.get("manifest_sha256"),
            "runs": runs,
        }
        (output / f"{dataset}_legacy_metrics.json").write_bytes(
            _json_bytes(payload)
        )


def _validate_manifest(manifest: dict) -> None:
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        raise ValueError("cleanup manifest 自校验失败。")
    if manifest.get("status") != "dry_run_validated_before_cleanup":
        raise ValueError("cleanup manifest 不是经过验证的 dry-run 状态。")


def _validate_manifest_files(manifest: dict) -> None:
    seen = set()
    for entry in manifest["entries"]:
        relative = entry["path"]
        if relative in seen:
            raise ValueError(f"cleanup manifest 路径重复：{relative}")
        seen.add(relative)
        source = PROJECT_ROOT / relative
        action = entry["action"]
        if action == "archive":
            target = PROJECT_ROOT / entry["destination"]
            if source.exists():
                raise ValueError(f"归档源仍存在，尚未完成移动：{relative}")
            if not target.is_file() or target.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"归档目标缺失或大小改变：{target}")
        elif action == "keep":
            if not source.is_file() or source.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"保留文件缺失或大小改变：{relative}")
        elif entry["category"] == "obsolete_config" and not source.exists():
            # 已跟踪 YAML 使用 apply_patch 删除，避免绕过代码审查。
            continue
        elif not source.is_file() or source.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"待删文件缺失或大小改变：{relative}")


def _remove_known_junction() -> None:
    junction = PROJECT_ROOT / "outputs/女声一小王子时间戳/node_modules"
    if not junction.exists():
        return
    attributes = getattr(
        junction.stat(follow_symlinks=False), "st_file_attributes", 0
    )
    if not attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise ValueError(f"预期临时 node_modules junction，实际不是 reparse point：{junction}")
    # rmdir 仅解除 junction 本身，不遍历或删除仓库外的目标目录。
    os.rmdir(junction)


def _remove_empty_directories() -> None:
    roots = [
        *(PROJECT_ROOT / relative for relative in CACHE_ROOTS),
        *(PROJECT_ROOT / relative for relative in DELETE_OUTPUT_ROOTS),
        *(PROJECT_ROOT / "outputs").glob("pytest_*"),
    ]
    config_roots = (
        PROJECT_ROOT / "configs/word_decoding/chineseeeg2_littleprince",
        PROJECT_ROOT / "configs/word_decoding/smn4lang",
        PROJECT_ROOT / "configs/word_decoding/libribrain100",
    )
    for root in (*roots, *config_roots):
        if not root.exists() or not root.is_dir():
            continue
        directories = sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            root.rmdir()
        except OSError:
            pass


def apply_cleanup(path: Path) -> dict:
    """严格按已冻结清单逐文件删除；不递归删除任何非空目录。"""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    _active_config_audit()
    _derived_audit()
    _validate_manifest_files(manifest)
    _remove_known_junction()
    removed_files = 0
    removed_bytes = 0
    for entry in manifest["entries"]:
        if entry["action"] != "delete":
            continue
        target = PROJECT_ROOT / entry["path"]
        if not target.exists():
            continue
        target.unlink()
        removed_files += 1
        removed_bytes += int(entry["size_bytes"])
    _remove_empty_directories()
    return {"removed_files": removed_files, "removed_bytes": removed_bytes}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="仅按现有 cleanup manifest 逐文件执行删除",
    )
    args = parser.parse_args(argv)
    path = PROJECT_ROOT / "experiments" / "cleanup_manifest.json"
    if args.apply:
        result = apply_cleanup(path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    manifest = build_cleanup_manifest()
    path.write_bytes(_json_bytes(manifest))
    legacy, _ = _legacy_outputs()
    write_legacy_summaries(legacy)
    print(
        json.dumps(
            {
                "manifest": _relative(path),
                "manifest_sha256": manifest["manifest_sha256"],
                "counts": manifest["counts"],
                "bytes_by_action": manifest["bytes_by_action"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
