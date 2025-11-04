"""Utilities for capturing intermediate activations."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def _to_tensor(output):
    if isinstance(output, torch.Tensor):
        if torch.isfinite(output).all():
            return output
        return torch.where(torch.isfinite(output), output, torch.zeros_like(output))
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                if torch.isfinite(item).all():
                    return item
                return torch.where(torch.isfinite(item), item, torch.zeros_like(item))
    raise TypeError("Unsupported module output type for feature capture")


class FeatureCatcher:
    def __init__(self, modules: Dict[str, nn.Module]) -> None:
        self.cache: Dict[str, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        for name, module in modules.items():
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name: str):
        def _hook(module, inputs, output):
            self.cache[name] = _to_tensor(output)

        return _hook

    def clear(self) -> None:
        self.cache.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
