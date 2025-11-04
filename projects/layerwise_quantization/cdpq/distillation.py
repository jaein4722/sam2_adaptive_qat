"""Distillation loss utilities for CDPQ training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from .features import FeatureCapture

__all__ = [
    "DistillationResult",
    "compute_layer_statistics_loss",
    "compute_feature_affinity_loss",
    "compute_distillation_loss",
]


@dataclass
class DistillationResult:
    total_loss: torch.Tensor
    layer_losses: Dict[str, torch.Tensor]
    teacher_features: Dict[str, List[FeatureCapture]]
    student_features: Dict[str, List[FeatureCapture]]


def compute_layer_statistics_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
) -> torch.Tensor:
    if teacher.shape != student.shape:
        student = F.adaptive_avg_pool2d(student, teacher.shape[-2:])
    t_mean = teacher.mean(dim=[2, 3])
    s_mean = student.mean(dim=[2, 3])
    t_var = teacher.var(dim=[2, 3])
    s_var = student.var(dim=[2, 3])
    return F.mse_loss(s_mean, t_mean) + F.mse_loss(s_var, t_var)


def compute_feature_affinity_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
) -> torch.Tensor:
    if teacher.shape != student.shape:
        student = F.adaptive_avg_pool2d(student, teacher.shape[-2:])
    b, c, h, w = teacher.shape
    t = teacher.view(b, c, h * w)
    s = student.view(b, c, h * w)
    t = F.normalize(t, dim=2)
    s = F.normalize(s, dim=2)
    affinity_teacher = torch.bmm(t, t.transpose(1, 2))
    affinity_cross = torch.bmm(t, s.transpose(1, 2))
    return F.mse_loss(affinity_cross, affinity_teacher)


def compute_distillation_loss(
    teacher_feats: Dict[str, List[FeatureCapture]],
    student_feats: Dict[str, List[FeatureCapture]],
    *,
    loss_type: str = "layer",
    layer_weights: Dict[str, float] | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    losses: Dict[str, torch.Tensor] = {}
    total = torch.tensor(0.0, device=next(iter(student_feats.values()))[0].tensor.device)

    for name, t_captures in teacher_feats.items():
        if name not in student_feats or not t_captures or not student_feats[name]:
            continue
        t_tensor = torch.stack([cap.tensor.to(total.device) for cap in t_captures], dim=0).mean(0)
        s_tensor = torch.stack([cap.tensor.to(total.device) for cap in student_feats[name]], dim=0).mean(0)
        if loss_type == "affinity":
            loss = compute_feature_affinity_loss(t_tensor, s_tensor)
        else:
            loss = compute_layer_statistics_loss(t_tensor, s_tensor)
        losses[name] = loss
        weight = layer_weights.get(name, 1.0) if layer_weights else 1.0
        total = total + weight * loss

    return total, losses
