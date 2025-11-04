"""Composite loss for adaptive QAT distillation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from ..utils import load_importance_config, resolve_map


def _safe_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(tensor).all():
        return tensor
    return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))


class AdaptiveQATLoss(nn.Module):
    def __init__(
        self,
        *,
        base_loss: nn.Module,
        layer_names: Optional[Iterable[str]] = None,
        importance: Optional[Mapping[str, float]] = None,
        importance_path: Optional[str] = None,
        kd_temperature: float = 1.0,
        base_weight: float = 1.0,
        layer_kd_weight: float = 1.0,
        bit_penalty_weight: float = 1.0,
        output_kd_weight: float = 1.0,
        alpha: float = 1.8,
        eps: float = 1e-3,
        budget: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.layer_names = list(layer_names) if layer_names is not None else None
        self.kd_temperature = kd_temperature
        self.base_weight = float(base_weight)
        self.layer_kd_weight = float(layer_kd_weight)
        self.bit_penalty_weight = float(bit_penalty_weight)
        self.output_kd_weight = float(output_kd_weight)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._progress = 0.0
        self._last_metrics: Dict[str, torch.Tensor] = {}

        budget_conf = dict(budget) if budget is not None else {}
        self._budget_enabled = bool(budget_conf.get("enabled", False))
        self._budget_target = float(budget_conf.get("target_avg_w_bits", 3.0))
        self._budget_warm_target = float(budget_conf.get("warm_target", self._budget_target))
        self._budget_mu_step = float(budget_conf.get("mu_step", 5e-3))
        self._budget_warmup_ratio = float(budget_conf.get("warmup_ratio", 0.0))
        mu_init = float(budget_conf.get("mu_init", 0.0))
        self.register_buffer("mu", torch.tensor(mu_init, dtype=torch.float32), persistent=True)

        file_importance: Dict[str, float] = {}
        if importance_path:
            file_importance = load_importance_config(importance_path)

        if self.layer_names is not None:
            resolved = resolve_map(
                self.layer_names,
                explicit=importance,
                default_map=file_importance,
                default_value=1.0,
            )
        else:
            source = importance or file_importance
            resolved = {k: float(v) for k, v in source.items()}
        self._importance_lookup = dict(resolved)
        if self.layer_names is not None:
            tensor = torch.tensor([resolved[name] for name in self.layer_names], dtype=torch.float32)
            self._importance_names = list(self.layer_names)
        else:
            self._importance_names = list(resolved.keys())
            tensor = torch.tensor([resolved[name] for name in self._importance_names], dtype=torch.float32)
        self.register_buffer("importance_tensor", tensor, persistent=False)

    def _ensure_layer_order(self, student_features: Mapping[str, torch.Tensor]) -> None:
        if self.layer_names is None:
            self.layer_names = list(student_features.keys())
            resolved = resolve_map(
                self.layer_names,
                default_map=self._importance_lookup,
                default_value=1.0,
            )
            self._importance_lookup = dict(resolved)
            tensor = torch.tensor(
                [resolved[name] for name in self.layer_names],
                dtype=torch.float32,
                device=self.importance_tensor.device,
            )
            self.importance_tensor = tensor
            self._importance_names = list(resolved.keys())

    def set_progress(self, progress: float) -> None:
        self._progress = float(progress)

    def get_latest_metrics(self) -> Dict[str, torch.Tensor]:
        return dict(self._last_metrics)

    def _extract_mask_sequence(self, stage: Mapping) -> Optional[Iterable[torch.Tensor]]:
        masks_high = stage.get("multistep_pred_masks_high_res")
        if masks_high is not None and len(masks_high) > 0:
            return masks_high
        masks_low = stage.get("multistep_pred_masks")
        if masks_low is not None and len(masks_low) > 0:
            return masks_low
        return None

    def _gather_logits(self, outputs) -> torch.Tensor:
        logits = []
        for stage in outputs:
            if not isinstance(stage, Mapping):
                continue
            masks = self._extract_mask_sequence(stage)
            if masks is None:
                continue
            mask = masks[-1]
            if mask.dim() > 2:
                height, width = mask.shape[-2:]
                mask = mask.reshape(-1, height, width).mean(dim=0)
            mask = _safe_tensor(mask)
            logits.append(mask)
        if not logits:
            raise ValueError("Unable to extract logits for KD output")
        stacked = torch.stack(logits, dim=0)
        return stacked.mean(dim=0)

    def forward(self, outputs, targets):  # type: ignore[override]
        student_outputs = outputs["student_outputs"]
        teacher_outputs = outputs["teacher_outputs"]
        student_features = outputs["student_features"]
        teacher_features = outputs["teacher_features"]
        bitwidths = outputs["bitwidths"]

        self._ensure_layer_order(student_features)
        assert self.layer_names is not None

        base_losses = self.base_loss(student_outputs, targets)
        if CORE_LOSS_KEY not in base_losses:
            raise ValueError("Underlying base loss must provide CORE_LOSS_KEY")
        base_core = base_losses[CORE_LOSS_KEY]

        device = base_core.device
        importance = self.importance_tensor.to(device)

        param_counts_map = outputs.get("param_counts")
        if param_counts_map is None:
            raise ValueError("Model outputs must include param_counts for adaptive loss.")
        assert self.layer_names is not None
        param_counts = torch.tensor(
            [float(param_counts_map[name]) for name in self.layer_names],
            dtype=torch.float32,
            device=device,
        )
        total_params = torch.clamp(param_counts.sum(), min=1.0)

        layer_kd = []
        layer_penalty = []
        bits_mean = []
        layer_bits = {}
        bit_tensors = []
        for idx, name in enumerate(self.layer_names):
            student_feat = _safe_tensor(student_features[name])
            teacher_feat = _safe_tensor(teacher_features[name]).detach()
            kd = F.mse_loss(student_feat, teacher_feat)
            bit = _safe_tensor(bitwidths[name]).to(device)
            scaled_bit = torch.pow(bit / (self.eps + importance[idx]), self.alpha)
            weight = param_counts[idx] / total_params
            penalty = weight * scaled_bit
            layer_kd.append(kd)
            layer_penalty.append(penalty)
            bits_mean.append(bit.detach())
            layer_bits[name] = bit.detach()
            bit_tensors.append(bit)

        layer_kd_sum = torch.stack(layer_kd).sum()
        bit_penalty_sum = torch.stack(layer_penalty).sum()
        layer_total = layer_kd_sum + bit_penalty_sum

        bits_stack = torch.stack(bit_tensors)
        avg_bits_w = torch.sum(param_counts * bits_stack) / total_params

        student_logits = _safe_tensor(self._gather_logits(student_outputs))
        teacher_logits = _safe_tensor(self._gather_logits(teacher_outputs)).detach()
        kd_output = F.mse_loss(student_logits, teacher_logits)

        budget_penalty = torch.tensor(0.0, device=device)
        target_bits_value = self._budget_target
        if self._budget_enabled:
            warm_active = self._progress < self._budget_warmup_ratio
            target_bits_value = self._budget_warm_target if warm_active else self._budget_target
            target_tensor = torch.tensor(target_bits_value, dtype=avg_bits_w.dtype, device=device)
            excess = torch.relu(avg_bits_w - target_tensor)
            budget_penalty = self.mu.to(device=device) * excess
        else:
            target_tensor = torch.tensor(target_bits_value, dtype=avg_bits_w.dtype, device=device)

        total_loss = (
            self.base_weight * base_core
            + self.layer_kd_weight * layer_kd_sum
            + self.bit_penalty_weight * bit_penalty_sum
            + self.output_kd_weight * kd_output
            + budget_penalty
        )

        if self._budget_enabled and self.training:
            with torch.no_grad():
                mu_next = torch.clamp(
                    self.mu + self._budget_mu_step * (avg_bits_w.detach() - target_tensor.detach()),
                    min=0.0,
                )
                self.mu.copy_(mu_next)

        losses = {k: v for k, v in base_losses.items()}
        losses["adaptive_layer_kd"] = layer_kd_sum.detach()
        losses["adaptive_bit_penalty"] = bit_penalty_sum.detach()
        losses["adaptive_layer_total"] = layer_total.detach()
        losses["adaptive_kd_output"] = kd_output.detach()
        losses["adaptive_bit_mean"] = torch.stack(bits_mean).mean()
        losses["adaptive_avg_bits_w"] = avg_bits_w.detach()
        losses["adaptive_budget_penalty"] = budget_penalty.detach()
        losses["adaptive_mu"] = self.mu.detach()
        losses["adaptive_layer_kd_scaled"] = (
            self.layer_kd_weight * layer_kd_sum
        ).detach()
        losses["adaptive_bit_penalty_scaled"] = (
            self.bit_penalty_weight * bit_penalty_sum
        ).detach()
        losses["adaptive_kd_output_scaled"] = (
            self.output_kd_weight * kd_output
        ).detach()
        losses["adaptive_base_scaled"] = (self.base_weight * base_core).detach()
        for name, bit_value in layer_bits.items():
            losses[f"adaptive_bits/{name}"] = bit_value
            losses[f"adaptive_param_count/{name}"] = torch.tensor(
                float(param_counts_map[name]), device=device
            )

        self._last_metrics = {
            "avg_bits_w": avg_bits_w.detach(),
            "mu": self.mu.detach().clone(),
            "bit_penalty": bit_penalty_sum.detach(),
            "budget_penalty": budget_penalty.detach(),
            "target_bits": torch.tensor(target_bits_value, device=device),
        }

        losses[CORE_LOSS_KEY] = total_loss
        return losses
