"""Helpers for resolving and iterating over named submodules."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch.nn as nn


def _get_child(module: nn.Module, name: str) -> nn.Module:
    if hasattr(module, name):
        child = getattr(module, name)
        if isinstance(child, nn.ModuleList):
            return child
        if isinstance(child, nn.ModuleDict):
            return child
        if not isinstance(child, nn.Module):
            raise AttributeError(f"Attribute '{name}' of {type(module)} is not an nn.Module")
        return child
    raise AttributeError(f"Module '{module.__class__.__name__}' has no attribute '{name}'")


def resolve_module(model: nn.Module, dotted_path: str) -> nn.Module:
    current: nn.Module | nn.ModuleList | nn.ModuleDict = model
    if not dotted_path:
        return model
    for part in dotted_path.split('.'):
        if isinstance(current, (list, tuple)):
            index = int(part)
            current = current[index]
            continue
        if isinstance(current, nn.ModuleList):
            index = int(part)
            current = current[index]
            continue
        if isinstance(current, nn.ModuleDict):
            current = current[part]
            continue
        current = _get_child(current, part)
    if not isinstance(current, nn.Module):
        raise AttributeError(f"Resolved object for '{dotted_path}' is not an nn.Module")
    return current


def iter_named_modules(model: nn.Module, names: Iterable[str]) -> List[Tuple[str, nn.Module]]:
    resolved: List[Tuple[str, nn.Module]] = []
    for name in names:
        resolved.append((name, resolve_module(model, name)))
    return resolved
