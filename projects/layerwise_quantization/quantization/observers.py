"""Observers used for calibration and confidence estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch

__all__ = [
    "ActivationRangeObserver",
    "collect_activation_ranges",
    "release_observers",
]

@dataclass
class ActivationRangeObserver:
    """Tracks running min/max statistics for tensors observed during calibration."""

    min_val: Optional[torch.Tensor] = None
    max_val: Optional[torch.Tensor] = None

    def update(self, tensor: torch.Tensor) -> None:
        with torch.no_grad():
            current_min = tensor.amin().detach().to("cpu")
            current_max = tensor.amax().detach().to("cpu")
            if self.min_val is None:
                self.min_val = current_min
                self.max_val = current_max
            else:
                self.min_val = torch.minimum(self.min_val, current_min)
                self.max_val = torch.maximum(self.max_val, current_max)

    def as_tuple(self) -> Tuple[float, float]:
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Observer has not seen any data yet.")
        return float(self.min_val.item()), float(self.max_val.item())

@torch.no_grad()
def collect_activation_ranges(
    model: torch.nn.Module,
    modules: Iterable[Tuple[str, torch.nn.Module]],
) -> Tuple[Dict[str, ActivationRangeObserver], Tuple[torch.utils.hooks.RemovableHandle, ...]]:
    """Attach forward hooks to capture activation ranges for target modules."""

    observers: Dict[str, ActivationRangeObserver] = {}
    handles = []

    for name, module in modules:
        observer = ActivationRangeObserver()

        def _hook(_, __, output, obs=observer):
            if isinstance(output, (tuple, list)):
                for candidate in output:
                    if isinstance(candidate, torch.Tensor):
                        obs.update(candidate)
                        break
            elif isinstance(output, torch.Tensor):
                obs.update(output)

        handle = module.register_forward_hook(_hook)
        observers[name] = observer
        handles.append(handle)

    return observers, tuple(handles)

@torch.no_grad()
def release_observers(handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()
