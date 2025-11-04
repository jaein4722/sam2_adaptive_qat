"""Quantized module wrappers for SAM 2 layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizers import quantize_tensor

__all__ = [
    "QuantizedModuleMixin",
    "QuantizedConv2d",
    "QuantizedLinear",
    "create_quantized_module",
]


@dataclass
class QuantizationConfig:
    num_bits: int = 8
    per_channel: bool = True
    channel_axis: int = 0


class QuantizedModuleMixin:
    """Mixin that adds dynamic weight quantization support."""

    def __init__(self, *, quant_config: Optional[QuantizationConfig] = None) -> None:
        self._quant_config = quant_config or QuantizationConfig()
        self.bit_width = self._quant_config.num_bits
        self.observed_error: float | None = None
        self._last_quantized_weight: Optional[torch.Tensor] = None

    @property
    def quantization_config(self) -> QuantizationConfig:
        return QuantizationConfig(
            num_bits=self.bit_width,
            per_channel=self._quant_config.per_channel,
            channel_axis=self._quant_config.channel_axis,
        )

    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.bit_width <= 0:
            return weight
        qconf = self.quantization_config
        qtensor = quantize_tensor(
            weight.detach(),
            qconf.num_bits,
            per_channel=qconf.per_channel,
            channel_axis=qconf.channel_axis,
        )
        dequantized = qtensor.dequantize().to(weight.device, dtype=weight.dtype)
        self._last_quantized_weight = dequantized
        return dequantized

    def load_quant_state(self, *, bit_width: int, error: Optional[float] = None) -> None:
        self.bit_width = bit_width
        self.observed_error = error

    def export_quant_state(self) -> dict:
        return {
            "bit_width": self.bit_width,
            "observed_error": self.observed_error,
        }


class QuantizedConv2d(QuantizedModuleMixin, nn.Conv2d):
    def __init__(self, conv: nn.Conv2d, *, quant_config: Optional[QuantizationConfig] = None) -> None:
        nn.Conv2d.__init__(
            self,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        self.load_state_dict(conv.state_dict())
        QuantizedModuleMixin.__init__(self, quant_config=quant_config)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quant_weight = self.quantize_weight(self.weight)
        return F.conv2d(
            input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedLinear(QuantizedModuleMixin, nn.Linear):
    def __init__(self, linear: nn.Linear, *, quant_config: Optional[QuantizationConfig] = None) -> None:
        nn.Linear.__init__(
            self,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.load_state_dict(linear.state_dict())
        QuantizedModuleMixin.__init__(self, quant_config=quant_config)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quant_weight = self.quantize_weight(self.weight)
        return F.linear(input, quant_weight, self.bias)


def create_quantized_module(module: nn.Module, *, quant_config: Optional[QuantizationConfig] = None) -> nn.Module:
    if isinstance(module, QuantizedModuleMixin):
        return module
    if isinstance(module, nn.Conv2d):
        return QuantizedConv2d(module, quant_config=quant_config)
    if isinstance(module, nn.Linear):
        return QuantizedLinear(module, quant_config=quant_config)
    return module
