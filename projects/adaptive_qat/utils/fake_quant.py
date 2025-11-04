"""Fake quantization utilities using straight-through estimators."""

from __future__ import annotations

import math
from typing import Iterable

import torch

_MIN_SCALE = 1e-6


def _sanitize_tensor(tensor: torch.Tensor, replacement: float) -> torch.Tensor:
    if torch.isfinite(tensor).all():
        return tensor
    fill = torch.full_like(tensor, replacement)
    return torch.where(torch.isfinite(tensor), tensor, fill)


class _FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        bit_width: torch.Tensor,
        max_val: torch.Tensor,
        allow_bit_grad: bool,
        grad_clip: float,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x

        x = _sanitize_tensor(x, 0.0)
        b = _sanitize_tensor(bit_width, 8.0)
        b = torch.clamp(b, min=1.0)

        max_val = _sanitize_tensor(max_val, 0.0)
        scalar_max = max_val.detach().abs()
        if scalar_max.numel() > 1:
            scalar_max = scalar_max.max()

        # Guard against degenerate ranges
        if scalar_max <= 0 or not torch.isfinite(scalar_max):
            return x

        qmax = torch.clamp((2 ** (b - 1.0)) - 1.0, min=1.0)
        qmin = -torch.pow(2.0, b - 1.0)
        scale = scalar_max / qmax
        scale = torch.clamp(scale, min=_MIN_SCALE)
        scale = _sanitize_tensor(scale, float(_MIN_SCALE))

        x_scaled = x / scale
        quant = torch.clamp(torch.round(x_scaled), qmin, qmax)
        quant = _sanitize_tensor(quant, 0.0)

        y = quant * scale

        ctx.save_for_backward(x, bit_width, scale, quant)
        ctx.bit_shape = bit_width.shape
        ctx.allow_bit_grad = bool(allow_bit_grad)
        ctx.grad_clip = float(grad_clip)
        ctx.numel = x.numel()
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Iterable[torch.Tensor]:  # type: ignore[override]
        grad_output = _sanitize_tensor(grad_output, 0.0)
        x, bit_width, scale, quant = ctx.saved_tensors

        grad_x = grad_output

        grad_bit = torch.zeros(ctx.bit_shape, device=grad_output.device, dtype=grad_output.dtype)
        if ctx.allow_bit_grad:
            x_scaled = x / scale
            diff = quant - x_scaled
            inner = (grad_output * diff).sum()
            grad_bit = inner * (-math.log(2.0) * scale)
            if ctx.numel > 0:
                grad_bit = grad_bit / float(ctx.numel)
            grad_bit = torch.clamp(grad_bit, min=-ctx.grad_clip, max=ctx.grad_clip)
            grad_bit = _sanitize_tensor(grad_bit, 0.0)
            grad_bit = grad_bit.reshape(ctx.bit_shape)

        grad_max = None
        return grad_x, grad_bit, grad_max, None, None


def fake_quantize(
    x: torch.Tensor,
    bit_width: torch.Tensor,
    max_val: torch.Tensor,
    allow_bit_grad: bool = True,
    grad_clip: float = 1.0,
) -> torch.Tensor:
    return _FakeQuantSTE.apply(x, bit_width, max_val, allow_bit_grad, grad_clip)
