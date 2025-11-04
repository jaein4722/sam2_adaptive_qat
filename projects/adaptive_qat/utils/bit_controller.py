"""Layer-wise bitwidth controllers for adaptive QAT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn


class StraightThroughRound(torch.autograd.Function):
    """Round with straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.round(value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad_output


@dataclass(frozen=True)
class BitRange:
    minimum: float = 2.0
    maximum: float = 8.0

    def clamp(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value, min=self.minimum, max=self.maximum)


class LayerBitController(nn.Module):
    """Maintains learnable bitwidths for a set of quantized layers."""

    def __init__(
        self,
        *,
        layer_names: Iterable[str],
        init_bits: Mapping[str, float],
        importance: Mapping[str, float],
        bit_range: BitRange,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()
        missing = set(layer_names) - set(init_bits)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing initial bitwidths for layers: {missing_str}")

        self.bit_range = bit_range
        self._bit_params = nn.ParameterDict()
        importance_values = torch.zeros(len(init_bits), dtype=torch.float32)
        self.register_buffer("_importance", importance_values, persistent=False)
        self._name_to_index: Dict[str, int] = {}
        importance_list = []
        for idx, name in enumerate(layer_names):
            init_value = torch.tensor(float(init_bits[name]), dtype=torch.float32)
            param = nn.Parameter(init_value, requires_grad=requires_grad)
            self._bit_params[name] = param
            self._name_to_index[name] = idx
            importance_list.append(float(importance.get(name, 1.0)))
        self._importance.copy_(torch.tensor(importance_list, dtype=torch.float32))

    def clamp_(self) -> None:
        """Clamp all bit parameters in-place to the configured range."""
        with torch.no_grad():
            for name, param in self._bit_params.items():
                param.copy_(self.bit_range.clamp(param))

    def importance(self, name: str) -> torch.Tensor:
        idx = self._name_to_index[name]
        return self._importance[idx]

    def importance_map(self) -> Dict[str, torch.Tensor]:
        return {name: self.importance(name) for name in self._bit_params.keys()}

    def parameter(self, name: str) -> nn.Parameter:
        if name not in self._bit_params:
            raise KeyError(f"Unknown bit parameter: {name}")
        return self._bit_params[name]

    def clamped_bits(self) -> Dict[str, torch.Tensor]:
        return {
            name: self.bit_range.clamp(param)
            for name, param in self._bit_params.items()
        }

    def rounded_bits(self) -> Dict[str, torch.Tensor]:
        return {
            name: StraightThroughRound.apply(self.bit_range.clamp(param))
            for name, param in self._bit_params.items()
        }

    def forward(self) -> Dict[str, torch.Tensor]:  # pragma: no cover - convenience wrapper
        return self.rounded_bits()

    def export_state(self) -> Dict[str, float]:
        return {
            name: float(self.bit_range.clamp(param.detach()).item())
            for name, param in self._bit_params.items()
        }

    @classmethod
    def from_defaults(
        cls,
        *,
        modules: Mapping[str, nn.Module],
        default_bits: float,
        importance: Optional[Mapping[str, float]] = None,
        bit_range: Optional[BitRange] = None,
        requires_grad: bool = True,
    ) -> "LayerBitController":
        names = list(modules.keys())
        init_bits = {name: float(default_bits) for name in names}
        importance = importance or {name: 1.0 for name in names}
        bit_range = bit_range or BitRange()
        return cls(
            layer_names=names,
            init_bits=init_bits,
            importance=importance,
            bit_range=bit_range,
            requires_grad=requires_grad,
        )

    def update_from_dict(self, values: Mapping[str, float]) -> None:
        with torch.no_grad():
            for name, value in values.items():
                if name not in self._bit_params:
                    continue
                self._bit_params[name].copy_(torch.tensor(float(value)))
            self.clamp_()

    def as_parameter_dict(self) -> nn.ParameterDict:
        return self._bit_params

    def to_metadata(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "bit": float(self.bit_range.clamp(param.detach()).item()),
                "importance": float(self.importance(name).item()),
            }
            for name, param in self._bit_params.items()
        }
