"""Helpers for discovering and manipulating quantizable layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .modules import QuantizationConfig, QuantizedModuleMixin, create_quantized_module
from .quantizers import QuantizedTensor

__all__ = [
    "ALLOWED_LAYER_TYPES",
    "QuantizationTarget",
    "discover_quantization_targets",
    "clone_layer_weights",
    "load_layer_weights",
    "apply_quantized_weight",
    "wrap_model_for_quantization",
    "collect_quantized_layers",
]
ALLOWED_LAYER_TYPES: Tuple[type, ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
)


@dataclass
class QuantizationTarget:
    """Metadata for a single quantizable layer."""

    name: str
    module: nn.Module
    num_bits: int = 8
    per_channel: bool = True
    channel_axis: int = 0
    observed_error: Optional[float] = None
    quantized: Optional[QuantizedTensor] = None
    notes: List[str] = field(default_factory=list)

    def add_note(self, message: str) -> None:
        self.notes.append(message)

def discover_quantization_targets(
    model: nn.Module,
    *,
    allowed_types: Sequence[type] = ALLOWED_LAYER_TYPES,
    skip_modules: Optional[Iterable[str]] = None,
    default_bits: int = 8,
    per_channel: bool = True,
) -> Dict[str, QuantizationTarget]:
    """Enumerate modules that can be quantized."""

    skip_set = set(skip_modules or [])
    targets: Dict[str, QuantizationTarget] = {}

    for name, module in model.named_modules():
        if any(name.startswith(prefix) for prefix in skip_set):
            continue
        if not isinstance(module, tuple(allowed_types)):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        channel_axis = 0
        if isinstance(module, nn.Linear):
            channel_axis = 0
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            channel_axis = 0

        targets[name] = QuantizationTarget(
            name=name,
            module=module,
            num_bits=default_bits,
            per_channel=per_channel,
            channel_axis=channel_axis,
        )

    return targets

def clone_layer_weights(targets: Dict[str, QuantizationTarget]) -> Dict[str, torch.Tensor]:
    return {
        name: target.module.weight.detach().cpu().clone()
        for name, target in targets.items()
    }


def load_layer_weights(
    targets: Dict[str, QuantizationTarget],
    cloned_weights: Dict[str, torch.Tensor],
) -> None:
    for name, tensor in cloned_weights.items():
        if name not in targets:
            continue
        module = targets[name].module
        module.weight.data.copy_(tensor.to(module.weight.device))

def apply_quantized_weight(
    target: QuantizationTarget,
    quantized: QuantizedTensor,
) -> torch.Tensor:
    dequantized = quantized.dequantize().to(target.module.weight.device)
    target.module.weight.data.copy_(dequantized)
    target.quantized = quantized
    return dequantized


def wrap_model_for_quantization(
    model: nn.Module,
    *,
    default_bits: int = 8,
    per_channel: bool = True,
    skip_prefixes: Optional[Iterable[str]] = None,
) -> Dict[str, QuantizedModuleMixin]:
    """Replace Conv/Linear layers with quantized variants."""

    skip = tuple(skip_prefixes or [])
    replacements: Dict[str, QuantizedModuleMixin] = {}
    modules = dict(model.named_modules())
    for name, module in list(modules.items()):
        if name in replacements:
            continue
        if name.endswith("."):
            continue
        if any(name.startswith(prefix) for prefix in skip):
            continue
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model if parent_name == "" else modules.get(parent_name)
        if parent is None:
            continue
        quant_module = create_quantized_module(
            module,
            quant_config=QuantizationConfig(num_bits=default_bits, per_channel=per_channel),
        )
        setattr(parent, child_name, quant_module)
        replacements[name] = quant_module
    return replacements


def collect_quantized_layers(model: nn.Module) -> Dict[str, QuantizedModuleMixin]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, QuantizedModuleMixin)
    }
