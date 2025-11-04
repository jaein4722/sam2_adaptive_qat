"""Utilities for loading per-layer importance configurations."""

from __future__ import annotations

import json
from typing import Dict, Iterable, Mapping, Optional

from iopath.common.file_io import g_pathmgr

_PARAM_SUFFIXES = {
    "weight",
    "bias",
    "running_mean",
    "running_var",
    "num_batches_tracked",
}


def _load_json(path: str) -> Mapping:
    with g_pathmgr.open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"Importance file {path} must contain a JSON object")
    return data


def _module_name(param_name: str) -> str:
    parts = param_name.split(".")
    if parts[-1] in _PARAM_SUFFIXES and len(parts) > 1:
        parts = parts[:-1]
    return ".".join(parts)


def _aggregate(values: Mapping[str, float]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    for name, value in values.items():
        module = _module_name(str(name))
        value = float(value)
        if module in aggregated:
            aggregated[module] = min(aggregated[module], value)
        else:
            aggregated[module] = value
    return aggregated


def load_importance_config(path: str) -> Dict[str, float]:
    """Return an importance map aggregated at module granularity."""
    data = _load_json(path)

    if "importance" in data:
        importance = _aggregate({str(k): float(v) for k, v in data["importance"].items()})
    else:
        importance = _aggregate({str(k): float(v) for k, v in data.items()})
    return importance


def resolve_map(
    layer_names: Iterable[str],
    *,
    explicit: Optional[Mapping[str, float]] = None,
    default_map: Optional[Mapping[str, float]] = None,
    default_value: float = 1.0,
) -> Dict[str, float]:
    resolved: Dict[str, float] = {}
    explicit = explicit or {}
    default_map = default_map or {}
    for name in layer_names:
        if name in explicit:
            resolved[name] = float(explicit[name])
        elif name in default_map:
            resolved[name] = float(default_map[name])
        else:
            resolved[name] = float(default_value)
    return resolved
