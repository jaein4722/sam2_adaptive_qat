"""Weight-only uniform quantization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    "QuantizedTensor",
    "dtype_for_bits",
    "symmetric_qparams",
    "quantize_tensor",
    "dequantize_tensor",
    "quantization_error",
]

@dataclass
class QuantizedTensor:
    """Container representing quantized values and their metadata."""

    values: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    num_bits: int
    symmetric: bool = True
    per_channel: bool = True
    channel_axis: int = 0

    def dequantize(self) -> torch.Tensor:
        return dequantize_tensor(
            self.values,
            self.scale,
            self.zero_point,
            channel_axis=self.channel_axis if self.per_channel else None,
        )

def dtype_for_bits(num_bits: int) -> torch.dtype:
    if num_bits <= 8:
        return torch.int8
    if num_bits <= 16:
        return torch.int16
    if num_bits <= 32:
        return torch.int32
    raise ValueError(f"Unsupported quantization bit-width: {num_bits}")

def symmetric_qparams(
    tensor: torch.Tensor,
    num_bits: int,
    *,
    per_channel: bool = True,
    channel_axis: int = 0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -1 << (num_bits - 1)
    data = tensor.detach()
    if per_channel:
        moved = data.movedim(channel_axis, 0).contiguous()
        flattened = moved.view(moved.size(0), -1)
        max_abs = flattened.abs().max(dim=1)[0]
    else:
        max_abs = data.abs().max()
    if torch.is_floating_point(max_abs):
        scale = torch.clamp(max_abs / max(qmax, 1), min=eps)
    else:
        scale = torch.clamp(max_abs.float() / max(qmax, 1), min=eps)
    zero_point = torch.zeros_like(scale)
    return scale, zero_point, qmin, qmax

def quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int,
    *,
    per_channel: bool = True,
    channel_axis: int = 0,
    symmetric: bool = True,
) -> QuantizedTensor:
    if not symmetric:
        raise NotImplementedError("Asymmetric quantization is not implemented yet.")
    scale, zero_point, qmin, qmax = symmetric_qparams(
        tensor,
        num_bits,
        per_channel=per_channel,
        channel_axis=channel_axis,
    )

    def _reshape(param: torch.Tensor) -> torch.Tensor:
        if not per_channel:
            return param
        shape = [1] * tensor.dim()
        shape[channel_axis] = tensor.shape[channel_axis]
        return param.view(shape)

    scale_reshaped = _reshape(scale.to(tensor.device))
    zero_point_reshaped = _reshape(zero_point.to(tensor.device))

    q_values = torch.clamp(
        torch.round(tensor / scale_reshaped) + zero_point_reshaped,
        qmin,
        qmax,
    )
    q_values = q_values.to(dtype_for_bits(num_bits)).cpu()

    return QuantizedTensor(
        values=q_values,
        scale=scale.cpu(),
        zero_point=zero_point.cpu(),
        num_bits=num_bits,
        symmetric=True,
        per_channel=per_channel,
        channel_axis=channel_axis,
    )

def dequantize_tensor(
    q_values: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    channel_axis: Optional[int] = 0,
) -> torch.Tensor:
    q = q_values.to(torch.float32)
    scale = scale.to(q.device)
    zero_point = zero_point.to(q.device)

    if channel_axis is not None and q.dim() > 0:
        shape = [1] * q.dim()
        shape[channel_axis] = q.shape[channel_axis]
        scale = scale.view(shape)
        zero_point = zero_point.view(shape)

    return (q - zero_point) * scale

def quantization_error(original: torch.Tensor, quantized: torch.Tensor) -> float:
    diff = (original - quantized).float().view(-1)
    denom = original.float().view(-1).norm() + 1e-12
    return float(diff.norm() / denom)
