"""Scalar metrics shared by structured training logs and tracking providers."""

import math
from collections.abc import Mapping
from typing import Any


INTERNAL_PROGRESS_METRICS = frozenset({
    "training/completed", "training/total", "training/progress",
    "training/restart_count", "training/elapsed_seconds",
})


def is_scalar(value: Any) -> bool:
    return not isinstance(value, bool) and (
        isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    )


def _public_name(name: Any) -> bool:
    return isinstance(name, str) and bool(name) and all(
        part and not part.startswith("_") for part in name.split("/")
    )


def recorded_scalar_metrics(
    row: Mapping[str, Any], aliases: Mapping[str, str]
) -> dict[str, float | int]:
    """Keep native metric names and compatibility aliases, without log metadata.

    Structured producers may emit flat scalar fields or a nested ``metrics``
    object. A slash separates nested metric namespaces. Strings, arrays, private
    fields and non-finite values are not scalar chart data.
    """
    metrics = {
        key: value for key, value in row.items()
        if _public_name(key) and is_scalar(value)
    }

    def collect(values: Mapping[str, Any], prefix: str = "") -> None:
        for key, value in values.items():
            if not _public_name(key):
                continue
            name = f"{prefix}/{key}" if prefix else key
            if is_scalar(value):
                # Explicit native names take precedence over flattened paths.
                metrics.setdefault(name, value)
        for key, value in values.items():
            if _public_name(key) and isinstance(value, Mapping):
                collect(value, f"{prefix}/{key}" if prefix else key)

    nested = row.get("metrics")
    if isinstance(nested, Mapping):
        collect(nested)
    for key, target in aliases.items():
        value = row.get(key)
        if _public_name(key) and _public_name(target) and is_scalar(value):
            metrics.setdefault(target, value)
    return metrics
