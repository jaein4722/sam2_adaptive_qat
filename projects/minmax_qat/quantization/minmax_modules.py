"""Simple min-max observers and fake-quant modules for SAM2 QAT."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MinMaxQuantConfig", "apply_minmax_quantization"]


_SUPPORTED_TYPES = (nn.Conv2d, nn.Linear)
_EPS = 1e-6


def _symmetric_qparams(num_bits: int) -> tuple[int, int]:
    if num_bits <= 1:
        raise ValueError("num_bits must be > 1 for symmetric quantization")
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -1 << (num_bits - 1)
    return qmin, qmax


@dataclass
class MinMaxQuantConfig:
    """Configuration for inserting min-max observers/fake-quant modules."""

    weight_bits: int = 8
    activation_bits: int = 8
    per_channel_weights: bool = True
    weight_channel_axis: int = 0
    activation_ema_decay: float = 0.99
    observer_epsilon: float = 1e-6
    quantize_inputs: bool = True
    quantize_outputs: bool = True
    target_modules: Optional[Sequence[str]] = None
    exclude_modules: Optional[Sequence[str]] = None

    def validate(self) -> None:
        if self.weight_bits <= 0 and self.activation_bits <= 0:
            raise ValueError("At least one of weight_bits/activation_bits must be > 0")
        if 0 < self.weight_bits <= 1:
            raise ValueError("weight_bits must be >= 2 when enabled")
        if 0 < self.activation_bits <= 1:
            raise ValueError("activation_bits must be >= 2 when enabled")
        if not 0.0 < self.activation_ema_decay < 1.0:
            raise ValueError("activation_ema_decay must be within (0, 1)")
        if self.observer_epsilon <= 0:
            raise ValueError("observer_epsilon must be positive")


class _ActivationFakeQuant(nn.Module):
    def __init__(self, num_bits: int, ema_decay: float, eps: float) -> None:
        super().__init__()
        self.num_bits = int(num_bits)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        if self.num_bits > 0:
            qmin, qmax = _symmetric_qparams(self.num_bits)
        else:
            qmin = qmax = 0
        self.qmin = qmin
        self.qmax = qmax
        self.register_buffer("max_abs", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_bits <= 0 or x.numel() == 0:
            return x
        with torch.no_grad():
            current = x.detach().abs().amax()
            if not torch.isfinite(current):
                current = torch.zeros_like(current)
            if bool(self.initialized):
                self.max_abs.mul_(self.ema_decay).add_(current * (1.0 - self.ema_decay))
            else:
                self.max_abs.copy_(current)
                self.initialized.fill_(True)
        max_val = torch.clamp(self.max_abs, min=self.eps)
        scale = torch.clamp(max_val / max(self.qmax, 1), min=self.eps)
        return torch.fake_quantize_per_tensor_affine(
            x,
            scale.to(x.device),
            0,
            self.qmin,
            self.qmax,
        )


class _WeightFakeQuant(nn.Module):
    def __init__(
        self,
        num_bits: int,
        *,
        per_channel: bool,
        channel_axis: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.num_bits = int(num_bits)
        self.per_channel = bool(per_channel)
        self.channel_axis = int(channel_axis)
        self.eps = float(eps)
        if self.num_bits > 0:
            self.qmin, self.qmax = _symmetric_qparams(self.num_bits)
        else:
            self.qmin = self.qmax = 0

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.num_bits <= 0 or weight.numel() == 0:
            return weight
        if self.per_channel:
            dims = list(range(weight.dim()))
            dims.remove(self.channel_axis)
            max_abs = weight.detach().abs().amax(dim=dims)
            max_abs = torch.clamp(max_abs, min=self.eps)
            scale = torch.clamp(max_abs / max(self.qmax, 1), min=self.eps).to(weight.device)
            zero_point = torch.zeros_like(scale, dtype=torch.int64, device=weight.device)
            return torch.fake_quantize_per_channel_affine(
                weight,
                scale,
                zero_point,
                self.channel_axis,
                self.qmin,
                self.qmax,
            )
        max_abs = torch.clamp(weight.detach().abs().amax(), min=self.eps)
        scale = torch.clamp(max_abs / max(self.qmax, 1), min=self.eps).to(weight.device)
        return torch.fake_quantize_per_tensor_affine(weight, scale, 0, self.qmin, self.qmax)


class _QuantizedConv2d(nn.Conv2d):
    def __init__(self, conv: nn.Conv2d, qconf: MinMaxQuantConfig) -> None:
        super().__init__(
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
        self.weight_quant = _WeightFakeQuant(
            qconf.weight_bits,
            per_channel=qconf.per_channel_weights,
            channel_axis=qconf.weight_channel_axis,
            eps=qconf.observer_epsilon,
        )
        self.act_in = _ActivationFakeQuant(
            qconf.activation_bits if qconf.quantize_inputs else 0,
            qconf.activation_ema_decay,
            qconf.observer_epsilon,
        )
        self.act_out = _ActivationFakeQuant(
            qconf.activation_bits if qconf.quantize_outputs else 0,
            qconf.activation_ema_decay,
            qconf.observer_epsilon,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        quant_input = self.act_in(input)
        quant_weight = self.weight_quant(self.weight)
        output = F.conv2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return self.act_out(output)


class _QuantizedLinear(nn.Linear):
    def __init__(self, linear: nn.Linear, qconf: MinMaxQuantConfig) -> None:
        super().__init__(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.load_state_dict(linear.state_dict())
        self.weight_quant = _WeightFakeQuant(
            qconf.weight_bits,
            per_channel=qconf.per_channel_weights,
            channel_axis=qconf.weight_channel_axis,
            eps=qconf.observer_epsilon,
        )
        self.act_in = _ActivationFakeQuant(
            qconf.activation_bits if qconf.quantize_inputs else 0,
            qconf.activation_ema_decay,
            qconf.observer_epsilon,
        )
        self.act_out = _ActivationFakeQuant(
            qconf.activation_bits if qconf.quantize_outputs else 0,
            qconf.activation_ema_decay,
            qconf.observer_epsilon,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        quant_input = self.act_in(input)
        quant_weight = self.weight_quant(self.weight)
        output = F.linear(quant_input, quant_weight, self.bias)
        return self.act_out(output)


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


def apply_minmax_quantization(model: nn.Module, config: MinMaxQuantConfig) -> List[str]:
    """Traverse the module tree and replace supported layers with quantized variants."""

    config.validate()
    target_patterns = tuple(config.target_modules or ())
    exclude_patterns = tuple(config.exclude_modules or ())
    replaced: List[str] = []

    def _should_quantize(module_name: str, module: nn.Module) -> bool:
        if not isinstance(module, _SUPPORTED_TYPES):
            return False
        if target_patterns and not _matches(module_name, target_patterns):
            return False
        if exclude_patterns and _matches(module_name, exclude_patterns):
            return False
        return True

    def _recursively_apply(root: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(root.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if _should_quantize(full_name, child):
                if isinstance(child, nn.Conv2d):
                    wrapped = _QuantizedConv2d(child, config)
                elif isinstance(child, nn.Linear):
                    wrapped = _QuantizedLinear(child, config)
                else:
                    continue
                setattr(root, child_name, wrapped)
                replaced.append(full_name)
                logging.info("[minmax_qat] Quantized module: %s", full_name)
            else:
                _recursively_apply(child, full_name)

    _recursively_apply(model)
    if not replaced:
        logging.warning("[minmax_qat] No modules were quantized. Check target/exclude patterns.")
    return replaced
