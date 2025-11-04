"""Loss functions for adaptive quantization with layerwise distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from ..utils.importance import resolve_importance_map


@dataclass
class LossWeights:
    ce: float = 1.0
    kd_output: float = 1.0
    bit_penalty: float = 1.0


class LayerwiseBitDistillationLoss(nn.Module):
    def __init__(
        self,
        *,
        layer_names: Iterable[str],
        importance: Mapping[str, float],
        kd_loss: str = "mse",
        temperature: float = 1.0,
        weights: LossWeights | None = None,
        importance_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layer_names = tuple(layer_names)
        if not self.layer_names:
            raise ValueError("LayerwiseBitDistillationLoss requires at least one layer")

        importance_values = [float(importance.get(name, 1.0)) for name in self.layer_names]
        self.register_buffer(
            "importance",
            torch.tensor(importance_values, dtype=torch.float32),
            persistent=False,
        )
        self.importance_eps = importance_eps
        self.temperature = temperature
        self.weights = weights or LossWeights()

        kd_loss = kd_loss.lower()
        if kd_loss == "mse":
            self.kd_fn = F.mse_loss
        elif kd_loss in {"smooth_l1", "huber"}:
            self.kd_fn = lambda input, target: F.smooth_l1_loss(input, target, beta=1.0)
        else:
            raise ValueError(f"Unsupported kd_loss '{kd_loss}'")
        self.ce_loss = nn.CrossEntropyLoss()

    def _compute_layer_losses(
        self,
        *,
        student_features: Mapping[str, torch.Tensor],
        teacher_features: Mapping[str, torch.Tensor],
        bitwidths: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats: Dict[str, torch.Tensor] = {}
        loss_accumulator = []
        kd_values = []
        penalty_values = []

        for idx, name in enumerate(self.layer_names):
            if name not in student_features:
                raise KeyError(f"Missing student feature '{name}'")
            if name not in teacher_features:
                raise KeyError(f"Missing teacher feature '{name}'")
            if name not in bitwidths:
                raise KeyError(f"Missing bitwidth for '{name}'")

            student_feat = student_features[name]
            teacher_feat = teacher_features[name].to(student_feat.device)
            bit = bitwidths[name].to(student_feat.device)
            kd = self.kd_fn(student_feat, teacher_feat)
            importance = self.importance[idx].to(student_feat.device)
            penalty = bit / (importance + self.importance_eps)
            layer_loss = kd + self.weights.bit_penalty * penalty
            loss_accumulator.append(layer_loss)
            kd_values.append(kd)
            penalty_values.append(penalty)

        total = torch.stack(loss_accumulator).sum()
        stats["kd_sum"] = torch.stack(kd_values).sum().detach()
        stats["penalty_sum"] = torch.stack(penalty_values).sum().detach()
        stats["num_layers"] = torch.tensor(len(self.layer_names), device=total.device)
        return total, stats

    def _kd_output_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        temp = self.temperature
        if temp <= 0:
            raise ValueError("temperature must be positive")
        student_log_prob = F.log_softmax(student_logits / temp, dim=1)
        teacher_prob = F.softmax(teacher_logits / temp, dim=1)
        kd = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        return kd * (temp ** 2)

    def forward(
        self,
        *,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: Optional[torch.Tensor],
        student_features: Mapping[str, torch.Tensor],
        teacher_features: Mapping[str, torch.Tensor],
        bitwidths: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        breakdown: Dict[str, torch.Tensor] = {}

        layer_loss, layer_stats = self._compute_layer_losses(
            student_features=student_features,
            teacher_features=teacher_features,
            bitwidths=bitwidths,
        )
        breakdown["layerwise"] = layer_loss.detach()
        breakdown.update(
            {
                "layer_kd_sum": layer_stats["kd_sum"],
                "layer_penalty_sum": layer_stats["penalty_sum"],
            }
        )

        ce_loss = torch.zeros((), device=student_logits.device)
        if targets is not None:
            ce_loss = self.ce_loss(student_logits, targets)
        breakdown["ce"] = ce_loss.detach()

        kd_output_loss = self._kd_output_loss(student_logits, teacher_logits)
        breakdown["kd_output"] = kd_output_loss.detach()

        total_loss = (
            self.weights.ce * ce_loss
            + self.weights.kd_output * kd_output_loss
            + layer_loss
        )
        breakdown["total"] = total_loss.detach()
        breakdown["bitwidth_mean"] = torch.stack(
            [bw.detach() for bw in bitwidths.values()]
        ).mean()

        return total_loss, breakdown


class QuantizationDistillationLoss(nn.Module):
    """Adapter loss used by the Trainer to compute total loss and log components."""

    def __init__(
        self,
        *,
        layer_names: Iterable[str],
        importance: Optional[Mapping[str, float]] = None,
        importance_path: Optional[str] = None,
        kd_loss: str = "mse",
        temperature: float = 1.0,
        weights: Optional[LossWeights] = None,
        importance_eps: float = 1e-6,
        student_head_key: Optional[str] = None,
        teacher_head_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.layer_names = tuple(layer_names)
        importance_map = resolve_importance_map(
            self.layer_names,
            importance=importance,
            importance_path=importance_path,
        )
        self.base_loss = LayerwiseBitDistillationLoss(
            layer_names=self.layer_names,
            importance=importance_map,
            kd_loss=kd_loss,
            temperature=temperature,
            weights=weights,
            importance_eps=importance_eps,
        )
        self.student_head_key = student_head_key
        self.teacher_head_key = teacher_head_key
        self.importance_map = importance_map

    def _select_head(self, output, key: Optional[str]):
        if key is None:
            return output
        if isinstance(output, Mapping):
            if key not in output:
                raise KeyError(f"Missing key '{key}' in output mapping")
            return output[key]
        if hasattr(output, key):
            return getattr(output, key)
        raise KeyError(f"Cannot extract '{key}' from output type {type(output)}")

    def forward(self, outputs: Mapping[str, torch.Tensor], targets: Optional[torch.Tensor]):
        student_logits = self._select_head(outputs["student_output"], self.student_head_key)
        teacher_logits = self._select_head(outputs["teacher_output"], self.teacher_head_key)
        student_features = outputs["student_features"]
        teacher_features = outputs["teacher_features"]
        bitwidths = outputs["bitwidths"]

        total_loss, breakdown = self.base_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            student_features=student_features,
            teacher_features=teacher_features,
            bitwidths=bitwidths,
        )

        return {
            CORE_LOSS_KEY: total_loss,
            "ce": breakdown["ce"],
            "kd_output": breakdown["kd_output"],
            "layerwise": breakdown["layerwise"],
            "layer_kd_sum": breakdown["layer_kd_sum"],
            "layer_penalty_sum": breakdown["layer_penalty_sum"],
            "bitwidth_mean": breakdown["bitwidth_mean"],
        }
