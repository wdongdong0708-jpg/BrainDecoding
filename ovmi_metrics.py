"""旧 OVMI 导入路径的兼容入口。

真实实现位于 :mod:`braindecoding.evaluation.ovmi`。
"""

from braindecoding.evaluation.ovmi import (
    _load_csv_reference,
    _ordered_labels,
    _reference_name,
    _result_template,
    _validated_reference,
    build_confusion_matrix,
    fixed_vocabulary_ovmi_metrics,
    full_ovmi_metrics,
    load_reference_distribution,
)

__all__ = [
    "build_confusion_matrix",
    "fixed_vocabulary_ovmi_metrics",
    "full_ovmi_metrics",
    "load_reference_distribution",
]
