"""Teacher/student assembly for adaptive QAT."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate

from ..utils import (
    ActivationQuantizer,
    BitRange,
    FeatureCatcher,
    LayerBitController,
    load_importance_config,
    resolve_map,
)
from ..utils.module_utils import iter_named_modules, resolve_module
from training.utils.checkpoint_utils import (
    load_checkpoint_and_apply_kernels,
    load_state_dict_into_model,
)
try:
    import torch.utils.checkpoint as torch_checkpoint
except ImportError:  # pragma: no cover - defensive guard
    torch_checkpoint = None


def checkpoint_wrapper(module: nn.Module, *, use_reentrant: bool = False) -> nn.Module:  # type: ignore
    if torch_checkpoint is None or not hasattr(torch_checkpoint, "checkpoint"):
        return module

    class _CheckpointModule(nn.Module):
        def __init__(self, wrapped: nn.Module) -> None:
            super().__init__()
            self.wrapped = wrapped

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.wrapped, name)

        def forward(self, *inputs, **kwargs):
            def _forward(*inner_inputs):
                return self.wrapped(*inner_inputs, **kwargs)

            try:
                return torch_checkpoint.checkpoint(
                    _forward,
                    *inputs,
                    use_reentrant=use_reentrant,
                )
            except TypeError:
                return torch_checkpoint.checkpoint(_forward, *inputs)

        def __len__(self):
            if hasattr(self.wrapped, "__len__"):
                return len(self.wrapped)  # type: ignore[arg-type]
            raise TypeError(f"Object of type {type(self.wrapped).__name__} has no len()")

        def __iter__(self):
            if hasattr(self.wrapped, "__iter__"):
                return iter(self.wrapped)
            raise TypeError(f"Object of type {type(self.wrapped).__name__} is not iterable")

        def __getitem__(self, idx):
            if hasattr(self.wrapped, "__getitem__"):
                return self.wrapped[idx]
            raise TypeError(f"Object of type {type(self.wrapped).__name__} does not support indexing")

    return _CheckpointModule(module)


@dataclass
class AdaptiveQATQuantConfig:
    layers: Optional[Iterable[str]] = None
    init_bits: Optional[Union[float, Mapping[str, float]]] = None
    min_bits: float = 2.0
    max_bits: float = 8.0
    requires_grad: bool = True
    importance: Optional[Mapping[str, float]] = None
    importance_path: Optional[str] = None
    smoothing: float = 1.0
    allow_bit_grad: bool = True
    smoothing_end_ratio: float = 0.0
    smoothing_importance_scale: bool = False
    act: Optional[Mapping[str, object]] = None
    checkpoint_modules: Optional[Iterable[str]] = None
    checkpoint_use_reentrant: bool = False


def _wrap_checkpoint_modules(
    root: nn.Module, module_names: Iterable[str], *, use_reentrant: bool
) -> None:
    for name in module_names:
        parts = name.split(".")
        parent = root
        try:
            for part in parts[:-1]:
                parent = getattr(parent, part)
            leaf_name = parts[-1]
            target = getattr(parent, leaf_name)
        except AttributeError:
            logging.warning("Checkpoint wrap skipped; module '%s' not found.", name)
            continue
        if isinstance(target, nn.Module):
            setattr(parent, leaf_name, checkpoint_wrapper(target, use_reentrant=use_reentrant))
        else:
            logging.warning("Checkpoint wrap skipped; object at '%s' is not a module.", name)


def _filter_module_names(model: nn.Module, names: Iterable[str]) -> Dict[str, nn.Module]:
    resolved = {}
    for name in names:
        try:
            module = resolve_module(model, name)
        except (AttributeError, IndexError, KeyError):
            continue
        resolved[name] = module
    return resolved


class AdaptiveQATModel(nn.Module):
    """Wraps teacher and student models, inserting quantization hooks on the student."""

    def __init__(
        self,
        *,
        teacher: Mapping | nn.Module,
        student: Mapping | nn.Module,
        quantization: AdaptiveQATQuantConfig,
        feature_layers: Optional[Iterable[str]] = None,
        teacher_checkpoint: Optional[str] = None,
        student_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__()

        teacher_module: nn.Module = (
            teacher if isinstance(teacher, nn.Module) else instantiate(teacher)
        )
        student_module: nn.Module = (
            student if isinstance(student, nn.Module) else instantiate(student)
        )

        if teacher_checkpoint:
            state = load_checkpoint_and_apply_kernels(
                checkpoint_path=teacher_checkpoint,
                ckpt_state_dict_keys=("model",),
            )
            load_state_dict_into_model(state, teacher_module, strict=True)
        if student_checkpoint:
            state = load_checkpoint_and_apply_kernels(
                checkpoint_path=student_checkpoint,
                ckpt_state_dict_keys=("model",),
            )
            load_state_dict_into_model(state, student_module, strict=True)

        for param in teacher_module.parameters():
            param.requires_grad_(False)
        teacher_module.eval()

        file_importance: Dict[str, float] = {}
        if quantization.importance_path:
            file_importance = load_importance_config(quantization.importance_path)

        candidate_layers = list(feature_layers) if feature_layers is not None else (
            list(quantization.layers) if quantization.layers is not None else list(file_importance.keys())
        )
        student_module_map = _filter_module_names(student_module, candidate_layers)
        teacher_module_map = _filter_module_names(teacher_module, candidate_layers)
        layer_names = sorted(set(student_module_map.keys()) & set(teacher_module_map.keys()))
        if not layer_names:
            raise ValueError(
                "No overlapping module names were found between teacher/student for quantization."
            )

        if quantization.init_bits is None:
            raise ValueError("quantization.init_bits must be provided as a scalar value.")

        if isinstance(quantization.init_bits, Mapping):
            raise ValueError("quantization.init_bits must be a scalar value; mappings are not supported.")

        try:
            init_bits_value = float(quantization.init_bits)
        except (TypeError, ValueError) as exc:
            raise ValueError("quantization.init_bits must be convertible to float.") from exc

        init_bits_map = {name: init_bits_value for name in layer_names}
        importance_map = resolve_map(
            layer_names,
            explicit=quantization.importance,
            default_map=file_importance,
            default_value=1.0,
        )
        missing_importance = set(layer_names) - set(importance_map.keys())
        if missing_importance:
            raise ValueError(f"Missing importance entries for layers: {sorted(missing_importance)}")
        extra_importance = set(importance_map.keys()) - set(layer_names)
        if extra_importance:
            logging.warning("Ignoring extra importance entries for layers: %s", sorted(extra_importance))

        stripped_lookup: Dict[str, str] = {}
        for name in layer_names:
            base = name
            for suffix in (".weight", ".bias"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
            if base in stripped_lookup and stripped_lookup[base] != name:
                logging.warning(
                    "Layer name collision after stripping weight/bias suffix: %s vs %s",
                    stripped_lookup[base],
                    name,
                )
            stripped_lookup[base] = name

        bit_controller = LayerBitController(
            layer_names=layer_names,
            init_bits=init_bits_map,
            bit_range=BitRange(quantization.min_bits, quantization.max_bits),
            requires_grad=quantization.requires_grad,
            smoothing=quantization.smoothing,
            importance=importance_map,
            smoothing_end_ratio=quantization.smoothing_end_ratio,
            smoothing_importance_scale=quantization.smoothing_importance_scale,
        )
        bit_param_ids = [id(bit_controller.parameter(name)) for name in layer_names]
        if len(bit_param_ids) != len(set(bit_param_ids)):
            raise AssertionError("Bitwidth parameters must be unique per layer.")

        student_modules = {name: student_module_map[name] for name in layer_names}
        teacher_modules = {name: teacher_module_map[name] for name in layer_names}
        param_counts = {
            name: float(sum(param.numel() for param in student_modules[name].parameters()))
            for name in layer_names
        }

        checkpoint_modules = list(quantization.checkpoint_modules or [])
        if checkpoint_modules:
            _wrap_checkpoint_modules(
                student_module,
                checkpoint_modules,
                use_reentrant=quantization.checkpoint_use_reentrant,
            )
        self._checkpoint_wrapped_modules = tuple(checkpoint_modules)

        self.teacher = teacher_module
        self.student = student_module
        self.bit_controller = bit_controller
        self.importance_map = importance_map
        self.layer_names = layer_names
        self.param_counts = param_counts

        self.student_quantizer = ActivationQuantizer(
            student_modules,
            bit_controller,
            allow_bit_grad=quantization.allow_bit_grad,
            act_config=quantization.act,
        )
        self.student_features = FeatureCatcher(student_modules)
        self.teacher_features = FeatureCatcher(teacher_modules)

    def sanitize_student_state_dict(self, state_dict: Mapping[str, torch.Tensor]):
        if not self._checkpoint_wrapped_modules:
            return state_dict
        sanitized = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            for module_name in self._checkpoint_wrapped_modules:
                for token in ("module", "wrapped"):
                    needle = f"{module_name}.{token}."
                    if needle in new_key:
                        new_key = new_key.replace(needle, f"{module_name}.")
                    suffix = f"{module_name}.{token}"
                    if new_key.endswith(suffix):
                        new_key = new_key[: -len(suffix)] + module_name
            sanitized[new_key] = value
        return sanitized

    def forward(self, batch):  # type: ignore[override]
        self.teacher_features.clear()
        self.student_features.clear()

        with torch.no_grad():
            teacher_out = self.teacher(batch)

        self.bit_controller.repair_()
        student_out = self.student(batch)

        student_features = {k: v for k, v in self.student_features.cache.items()}
        teacher_features = {k: v for k, v in self.teacher_features.cache.items()}
        bitwidths = self.bit_controller.bits()

        return {
            "student_outputs": student_out,
            "teacher_outputs": teacher_out,
            "student_features": student_features,
            "teacher_features": teacher_features,
            "bitwidths": bitwidths,
            "param_counts": self.param_counts,
        }
