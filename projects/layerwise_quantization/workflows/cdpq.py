"""Confidence-Driven Progressive Quantization workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from ..io.storage import serialize_targets, save_package
from ..quantization.quantizers import quantization_error, quantize_tensor
from ..quantization.utils import (
    QuantizationTarget,
    apply_quantized_weight,
    discover_quantization_targets,
)
from ..adapters import build_sam2_model, ensure_device
from .calibration import CalibrationConfig, build_calibration_loader, calibrate_model
from .ptq import PTQConfig

__all__ = ["CDPQConfig", "run_cdpq"]

@dataclass
class CDPQConfig(PTQConfig):
    bit_candidates: Sequence[int] = (8, 6, 4)
    confidence_threshold: float = 0.03
    min_bits: Optional[int] = None
    max_bits: Optional[int] = None

def run_cdpq(config: CDPQConfig):
    device = ensure_device(config.device)
    model = build_sam2_model(
        config_name=config.config_name,
        checkpoint_path=config.checkpoint_path,
        device=device,
        mode="eval",
    )

    targets = discover_quantization_targets(
        model,
        skip_modules=config.skip_prefixes,
        default_bits=config.bits,
        per_channel=config.per_channel,
    )

    loader = build_calibration_loader(model, config.calibration)
    activation_ranges = calibrate_model(
        model,
        loader,
        targets,
        device=device,
        max_batches=config.calibration.max_batches,
    )

    candidates = sorted(set(config.bit_candidates), reverse=True)
    if config.max_bits is not None:
        candidates = [b for b in candidates if b <= config.max_bits]
    if config.min_bits is not None:
        candidates = [b for b in candidates if b >= config.min_bits]
    if not candidates:
        raise ValueError("No candidate bit-widths remain after filtering.")

    threshold = float(config.confidence_threshold)

    for name, target in targets.items():
        original_weight = target.module.weight.detach().cpu()
        best_bits = candidates[0]
        best_qtensor = quantize_tensor(
            original_weight,
            best_bits,
            per_channel=target.per_channel,
            channel_axis=target.channel_axis,
        )
        best_error = quantization_error(
            original_weight,
            best_qtensor.dequantize(),
        )
        target.notes.append(
            f"Baseline {best_bits}-bit error={best_error:.4f}"
        )

        for bits in candidates[1:]:
            candidate_q = quantize_tensor(
                original_weight,
                bits,
                per_channel=target.per_channel,
                channel_axis=target.channel_axis,
            )
            candidate_error = quantization_error(
                original_weight,
                candidate_q.dequantize(),
            )
            if candidate_error <= threshold:
                best_bits = bits
                best_qtensor = candidate_q
                best_error = candidate_error
                target.notes.append(
                    f"Accepted {bits}-bit (error={candidate_error:.4f} <= {threshold:.4f})"
                )
            else:
                target.notes.append(
                    f"Stopped at {bits}-bit (error={candidate_error:.4f} > {threshold:.4f})"
                )
                break

        target.num_bits = best_bits
        dequantized = apply_quantized_weight(target, best_qtensor)
        target.observed_error = best_error

    metadata = dict(config.metadata)
    metadata.update(
        {
            "bit_candidates": list(candidates),
            "confidence_threshold": threshold,
            "activation_ranges": activation_ranges,
        }
    )

    package = serialize_targets(
        targets,
        model_config=config.config_name,
        checkpoint_path=config.checkpoint_path,
        metadata=metadata,
    )

    if config.output_path:
        save_package(package, Path(config.output_path))

    return package, targets, activation_ranges
