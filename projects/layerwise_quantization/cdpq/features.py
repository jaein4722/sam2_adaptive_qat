"""Utilities for extracting intermediate features from SAM 2 models."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

import torch
import torch.nn as nn

__all__ = ["FeatureHook", "FeatureCapture", "capture_features"]


@dataclass
class FeatureCapture:
    name: str
    tensor: torch.Tensor


class FeatureHook:
    def __init__(self, module: nn.Module, name: str, *, flatten: bool = False, detach: bool = True) -> None:
        self.module = module
        self.name = name
        self.flatten = flatten
        self.detach = detach
        self.outputs: List[FeatureCapture] = []
        self._handle = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, _input, output) -> None:
        if isinstance(output, (tuple, list)):
            tensor = output[0]
        else:
            tensor = output
        if not isinstance(tensor, torch.Tensor):
            return
        capture = tensor.detach() if self.detach else tensor
        if self.flatten:
            capture = capture.flatten(1)
        self.outputs.append(FeatureCapture(self.name, capture))

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass


@contextmanager
def capture_features(
    model: nn.Module,
    module_names: Iterable[str],
    *,
    flatten: bool = False,
    detach: bool = True,
) -> Iterator[Dict[str, List[FeatureCapture]]]:
    hooks: Dict[str, FeatureHook] = {}
    modules = dict(model.named_modules())
    for name in module_names:
        module = modules.get(name)
        if module is None:
            continue
        hooks[name] = FeatureHook(module, name=name, flatten=flatten, detach=detach)
    try:
        yield {name: hook.outputs for name, hook in hooks.items()}
    finally:
        for hook in hooks.values():
            hook.close()
