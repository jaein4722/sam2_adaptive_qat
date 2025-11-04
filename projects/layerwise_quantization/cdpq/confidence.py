"""Confidence heuristics for bit-width reduction."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

def _filter_layers(
    bitwidth_lookup: Dict[str, int],
    candidates: Iterable[str],
    *,
    min_bits: int,
) -> List[str]:
    result: List[str] = []
    for name in candidates:
        bit = bitwidth_lookup.get(name)
        if bit is None:
            continue
        if bit > min_bits:
            result.append(name)
    return result


def confidence_baseline(
    layer_losses: Dict[str, List[float]],
    bitwidth_lookup: Dict[str, int],
    *,
    threshold: float,
    min_bits: int,
) -> List[str]:
    keep: List[str] = []
    for name, losses in layer_losses.items():
        if not losses:
            continue
        if np.mean(losses) <= threshold:
            keep.append(name)
    return _filter_layers(bitwidth_lookup, keep, min_bits=min_bits)


def confidence_variance(
    layer_losses: Dict[str, List[float]],
    bitwidth_lookup: Dict[str, int],
    *,
    mean_threshold: float,
    var_threshold: float,
    min_bits: int,
) -> List[str]:
    keep: List[str] = []
    for name, losses in layer_losses.items():
        if len(losses) < 3:
            continue
        arr = np.array(losses)
        if arr.mean() <= mean_threshold and arr.var() <= var_threshold:
            keep.append(name)
    return _filter_layers(bitwidth_lookup, keep, min_bits=min_bits)
