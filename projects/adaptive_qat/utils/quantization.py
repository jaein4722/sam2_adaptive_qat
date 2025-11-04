"""Bitwidth controller and activation quantization hooks."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fake_quant import fake_quantize


@dataclass(frozen=True)
class BitRange:
    minimum: float = 2.0
    maximum: float = 8.0

    def clamp(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value, min=self.minimum, max=self.maximum)


class LayerBitController(nn.Module):
    def __init__(
        self,
        *,
        layer_names: Iterable[str],
        init_bits: Dict[str, float],
        bit_range: BitRange,
        requires_grad: bool = True,
        smoothing: float = 1.0,
        importance: Optional[Mapping[str, float]] = None,
        smoothing_end_ratio: float = 0.0,
        smoothing_importance_scale: bool = False,
    ) -> None:
        super().__init__()
        self.bit_range = bit_range
        if smoothing <= 0.0 or smoothing > 1.0:
            raise ValueError("smoothing must be within (0, 1].")
        if smoothing_end_ratio < 0.0 or smoothing_end_ratio > 1.0:
            raise ValueError("smoothing_end_ratio must be within [0, 1].")

        self._module_names: List[str] = list(layer_names)
        self._base_smoothing = float(smoothing)
        self._smoothing_end_ratio = float(smoothing_end_ratio)
        self._smoothing_importance_scale = bool(smoothing_importance_scale)

        importance_tensor = torch.tensor(
            [float(importance.get(name, 1.0)) if importance else 1.0 for name in self._module_names],
            dtype=torch.float32,
        )
        self.register_buffer("_importance", importance_tensor, persistent=False)

        self._param_names: List[str] = []
        for idx, name in enumerate(self._module_names):
            param_name = name.replace(".", "_DOT_")
            init = torch.tensor(float(init_bits.get(name, bit_range.maximum)), dtype=torch.float32)
            self.register_parameter(param_name, nn.Parameter(init, requires_grad=requires_grad))
            self._param_names.append(param_name)
        last_bits = [float(init_bits.get(name, bit_range.maximum)) for name in self._module_names]
        self.register_buffer("_last_bits", torch.tensor(last_bits, dtype=torch.float32), persistent=False)

        smoothing_scales = self._compute_importance_scaling()
        smoothing_values = torch.clamp(self._base_smoothing * smoothing_scales, min=0.0, max=1.0)
        self.register_buffer("_smoothing_values", smoothing_values.clone(), persistent=False)
        self.register_buffer("_current_smoothing", smoothing_values.clone(), persistent=False)

    @property
    def module_names(self) -> List[str]:
        return list(self._module_names)

    def _compute_importance_scaling(self) -> torch.Tensor:
        if not self._smoothing_importance_scale or self._importance.numel() == 0:
            return torch.ones_like(self._importance)
        vals = self._importance
        variance = torch.var(vals)
        if variance <= 1e-6:
            return torch.ones_like(vals)
        normalized = (vals - vals.mean()) / torch.sqrt(variance + 1e-6)
        scale = 1.0 + 0.2 * normalized
        return torch.clamp(scale, min=0.6, max=1.2)

    def _sanitize(self, value: torch.Tensor) -> torch.Tensor:
        if torch.isfinite(value).all():
            return self.bit_range.clamp(value)
        replacement = torch.full_like(value, self.bit_range.maximum)
        sanitized = torch.where(torch.isfinite(value), value, replacement)
        return self.bit_range.clamp(sanitized)

    def sanitize(self, value: torch.Tensor) -> torch.Tensor:
        return self._sanitize(value)

    def _param_name(self, name: str) -> str:
        return name.replace('.', '_DOT_')

    def repair_(self) -> None:
        with torch.no_grad():
            for idx, param_name in enumerate(self._param_names):
                param = getattr(self, param_name)
                sanitized = self._sanitize(param)
                smoothing = float(self._current_smoothing[idx].item())
                if smoothing < 1.0:
                    smoothed = torch.lerp(self._last_bits[idx], sanitized, smoothing)
                    param.copy_(smoothed)
                    self._last_bits[idx] = smoothed.detach()
                else:
                    if not torch.equal(sanitized, param):
                        param.copy_(sanitized)
                    self._last_bits[idx] = sanitized.detach()

    def clamp_(self) -> None:
        with torch.no_grad():
            for idx, param_name in enumerate(self._param_names):
                param = getattr(self, param_name)
                sanitized = self._sanitize(param)
                param.copy_(sanitized)
                self._last_bits[idx] = sanitized.detach()

    def parameter(self, name: str) -> nn.Parameter:
        return getattr(self, self._param_name(name))

    def bits(self) -> Dict[str, torch.Tensor]:
        mapped = {}
        for idx, (full_name, param) in enumerate(self.named_parameters()):
            module_name = full_name.replace('_DOT_', '.')
            mapped[module_name] = self._sanitize(param)
            self._last_bits[idx] = mapped[module_name].detach()
        return mapped

    def forward(self) -> Dict[str, torch.Tensor]:  # pragma: no cover
        return self.bits()

    def set_smoothing(self, smoothing: float) -> None:
        if smoothing <= 0.0 or smoothing > 1.0:
            raise ValueError("smoothing must be within (0, 1].")
        self._base_smoothing = float(smoothing)
        scales = self._compute_importance_scaling()
        updated = torch.clamp(self._base_smoothing * scales, min=0.0, max=1.0)
        with torch.no_grad():
            self._smoothing_values.copy_(updated)
            self._current_smoothing.copy_(self._smoothing_values)

    def update_progress(self, progress: float) -> None:
        progress = float(progress)
        target = torch.ones_like(self._current_smoothing) if progress >= self._smoothing_end_ratio else self._smoothing_values
        with torch.no_grad():
            self._current_smoothing.copy_(target)


@dataclass
class ActivationQuantConfig:
    ema_scale: bool = False
    ema_decay: float = 0.95
    qdrop_p: float = 0.0
    pact: bool = False
    pact_init: float = 10.0

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, object]]) -> "ActivationQuantConfig":
        if config is None:
            return cls()
        valid = {field.name for field in fields(cls)}
        filtered = {k: config[k] for k in config if k in valid}
        return cls(**filtered)


@dataclass
class _LayerState:
    bit_param: nn.Parameter
    ema_index: int
    pact_key: Optional[str]


class ActivationQuantizer(nn.Module):
    def __init__(
        self,
        modules: Dict[str, nn.Module],
        controller: LayerBitController,
        *,
        allow_bit_grad: bool = True,
        bit_grad_clip: float = 1.0,
        act_config: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._controller = controller
        self._allow_bit_grad = bool(allow_bit_grad)
        self._bit_grad_clip = float(bit_grad_clip)
        self._states: Dict[str, _LayerState] = {}
        self._act = ActivationQuantConfig.from_mapping(act_config)

        module_names = list(modules.keys())
        self.register_buffer(
            "ema_scales",
            torch.zeros(len(module_names), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ema_initialized",
            torch.zeros(len(module_names), dtype=torch.bool),
            persistent=False,
        )
        self._pact_alpha = nn.ParameterDict()

        for idx, name in enumerate(module_names):
            bit_param = controller.parameter(name)
            pact_key = None
            if self._act.pact:
                pact_key = self._sanitize_name(name)
                self._pact_alpha[pact_key] = nn.Parameter(
                    torch.tensor(float(self._act.pact_init), dtype=torch.float32)
                )
            self._states[name] = _LayerState(bit_param=bit_param, ema_index=idx, pact_key=pact_key)
            handle = modules[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace(".", "_DOT_")

    def _make_hook(self, name: str):
        state = self._states[name]

        def _hook(module, inputs, output):
            bit_value = self._controller.sanitize(state.bit_param)
            return self._quantize_output(output, bit_value, state, module.training)

        return _hook

    def _quantize_output(
        self,
        output,
        bit_value: torch.Tensor,
        state: _LayerState,
        module_training: bool,
    ):
        if isinstance(output, torch.Tensor):
            return self._quantize_tensor(output, bit_value, state, module_training)
        if isinstance(output, (tuple, list)):
            quantized = [
                self._quantize_output(item, bit_value, state, module_training) if isinstance(item, torch.Tensor) else item
                for item in output
            ]
            return type(output)(quantized)
        if isinstance(output, dict):
            return {
                key: self._quantize_output(val, bit_value, state, module_training) if isinstance(val, torch.Tensor) else val
                for key, val in output.items()
            }
        return output

    def _quantize_tensor(
        self,
        tensor: torch.Tensor,
        bit_value: torch.Tensor,
        state: _LayerState,
        module_training: bool,
    ) -> torch.Tensor:
        if tensor.numel() == 0 or not tensor.is_floating_point():
            return tensor

        quant_input = tensor
        if self._act.pact and state.pact_key is not None:
            alpha_raw = self._pact_alpha[state.pact_key]
            alpha = F.softplus(alpha_raw)
            quant_input = torch.clamp(quant_input, min=-alpha, max=alpha)

        max_val = quant_input.detach().abs().amax()

        if self._act.ema_scale:
            idx = state.ema_index
            with torch.no_grad():
                initialized = bool(self.ema_initialized[idx])
                if not initialized:
                    updated = max_val
                    self.ema_initialized[idx] = torch.tensor(
                        True, dtype=torch.bool, device=self.ema_initialized.device
                    )
                else:
                    updated = self.ema_scales[idx] * self._act.ema_decay + max_val * (1.0 - self._act.ema_decay)
                self.ema_scales[idx] = updated
            max_val = self.ema_scales[idx].detach()

        if not torch.isfinite(max_val):
            return quant_input

        max_val = torch.clamp(max_val, min=1e-6)

        if self._act.qdrop_p > 0.0 and (self.training or module_training):
            if torch.rand(1, device=quant_input.device).item() < self._act.qdrop_p:
                return quant_input

        max_tensor = torch.as_tensor(max_val, dtype=bit_value.dtype, device=bit_value.device)
        return fake_quantize(
            quant_input,
            bit_value,
            max_tensor,
            allow_bit_grad=self._allow_bit_grad,
            grad_clip=self._bit_grad_clip,
        )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
