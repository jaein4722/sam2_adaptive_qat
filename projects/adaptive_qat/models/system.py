"""Teacher/student assembly for adaptive QAT."""

from __future__ import annotations

import logging
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


def _filter_module_names(model: nn.Module, names: Iterable[str]) -> Dict[str, nn.Module]:
    resolved = {}
    for name in names:
        try:
            module = resolve_module(model, name)
        except AttributeError:
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
