"""Serialization helpers for quantized SAM 2 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import torch

from ..quantization.modules import QuantizedModuleMixin
from ..quantization.quantizers import QuantizedTensor, quantize_tensor
from ..quantization.utils import QuantizationTarget

__all__ = [
    "QuantizedLayerState",
    "QuantizedModelPackage",
    "serialize_targets",
    "deserialize_targets",
    "save_package",
    "load_package",
]

@dataclass
class QuantizedLayerState:
    name: str
    tensor: QuantizedTensor
    error: Optional[float] = None

    def as_dict(self) -> Dict[str, torch.Tensor | int | bool | float]:
        return {
            "num_bits": self.tensor.num_bits,
            "per_channel": self.tensor.per_channel,
            "channel_axis": self.tensor.channel_axis,
            "symmetric": self.tensor.symmetric,
            "values": self.tensor.values,
            "scale": self.tensor.scale,
            "zero_point": self.tensor.zero_point,
            "error": self.error,
        }

@dataclass
class QuantizedModelPackage:
    format_version: str = "1.0"
    model_config: Optional[str] = None
    checkpoint_path: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    layers: Dict[str, QuantizedLayerState] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        return {
            "format_version": self.format_version,
            "model_config": self.model_config,
            "checkpoint_path": self.checkpoint_path,
            "num_layers": len(self.layers),
        }

SerializableTarget = Union[QuantizedModuleMixin, QuantizationTarget]


def serialize_targets(
    modules: Mapping[str, SerializableTarget],
    *,
    model_config: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> QuantizedModelPackage:
    package = QuantizedModelPackage(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        metadata=dict(metadata or {}),
    )
    for name, module in modules.items():
        if isinstance(module, QuantizedModuleMixin):
            weight = module.weight.detach().cpu()
            qconf = module.quantization_config
            qtensor = quantize_tensor(
                weight,
                qconf.num_bits,
                per_channel=qconf.per_channel,
                channel_axis=qconf.channel_axis,
            )
            error = module.observed_error
        elif isinstance(module, QuantizationTarget):
            weight = module.module.weight.detach().cpu()
            qtensor = module.quantized
            if qtensor is None:
                qtensor = quantize_tensor(
                    weight,
                    module.num_bits,
                    per_channel=module.per_channel,
                    channel_axis=module.channel_axis,
                )
            error = module.observed_error
        else:
            raise TypeError(
                f"Unsupported module type for serialization: {type(module).__name__}"
            )

        package.layers[name] = QuantizedLayerState(
            name=name,
            tensor=qtensor,
            error=error,
        )
    return package

def deserialize_targets(package: QuantizedModelPackage) -> Dict[str, QuantizedTensor]:
    tensors: Dict[str, QuantizedTensor] = {}
    for name, state in package.layers.items():
        tensors[name] = state.tensor
    return tensors


def save_package(package: QuantizedModelPackage, path: str | Path) -> None:
    serializable = {
        "format_version": package.format_version,
        "model_config": package.model_config,
        "checkpoint_path": package.checkpoint_path,
        "metadata": package.metadata,
        "layers": {name: state.as_dict() for name, state in package.layers.items()},
    }
    torch.save(serializable, Path(path))


def load_package(path: str | Path) -> QuantizedModelPackage:
    raw = torch.load(Path(path), map_location="cpu")
    package = QuantizedModelPackage(
        format_version=raw.get("format_version", "1.0"),
        model_config=raw.get("model_config"),
        checkpoint_path=raw.get("checkpoint_path"),
        metadata=dict(raw.get("metadata", {})),
    )
    for name, payload in raw.get("layers", {}).items():
        tensor = QuantizedTensor(
            values=payload["values"],
            scale=payload["scale"],
            zero_point=payload["zero_point"],
            num_bits=int(payload["num_bits"]),
            symmetric=bool(payload.get("symmetric", True)),
            per_channel=bool(payload.get("per_channel", True)),
            channel_axis=int(payload.get("channel_axis", 0)),
        )
        package.layers[name] = QuantizedLayerState(
            name=name,
            tensor=tensor,
            error=payload.get("error"),
        )
    return package
