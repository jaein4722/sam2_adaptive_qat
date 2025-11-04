"""Post-Training Quantization workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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

__all__ = ["PTQConfig", "run_ptq"]

@dataclass
class PTQConfig:
    config_name: str
    checkpoint_path: Optional[str] = None
    bits: int = 8
    per_channel: bool = True
    device: Optional[str | torch.device] = None
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    skip_prefixes: Sequence[str] = ()
    output_path: Optional[str] = None
    package_output_path: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)

def _default_package_path(checkpoint_path: Path) -> Path:
    stem = checkpoint_path.stem
    return checkpoint_path.with_name(f"{stem}_quantized_package.pt")

def run_ptq(config: PTQConfig):
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

    output_path: Optional[Path] = None
    package_output_path: Optional[Path] = None
    if config.output_path:
        output_path = Path(config.output_path).expanduser()
    if config.package_output_path:
        package_output_path = Path(config.package_output_path).expanduser()
    if package_output_path is None and output_path is not None:
        package_output_path = _default_package_path(output_path)

    raw_metadata = config.metadata or {}
    requested_per_layer_bits = raw_metadata.get("per_layer_bits")
    per_layer_bits: Dict[str, int] = {}
    if isinstance(requested_per_layer_bits, dict):
        for key, value in requested_per_layer_bits.items():
            try:
                per_layer_bits[str(key)] = int(value)
            except (TypeError, ValueError):
                print(
                    f"Warning: ignoring non-integer bit override for '{key}': {value}"
                )
    elif requested_per_layer_bits is not None:
        print(
            "Warning: expected 'per_layer_bits' metadata to be a dictionary; "
            f"got {type(requested_per_layer_bits).__name__}"
        )

    applied_overrides: Dict[str, int] = {}
    unmatched_prefixes = set(per_layer_bits.keys())
    if per_layer_bits:
        # Prefer the most specific prefixes first.
        sorted_overrides: Sequence[Tuple[str, int]] = sorted(
            per_layer_bits.items(), key=lambda item: len(item[0]), reverse=True
        )
        for name, target in targets.items():
            for prefix, bits in sorted_overrides:
                if name == prefix or name.startswith(f"{prefix}."):
                    target.num_bits = bits
                    applied_overrides[name] = bits
                    unmatched_prefixes.discard(prefix)
                    target.add_note(f"bit override set to {bits} via '{prefix}'")
                    break
        if unmatched_prefixes:
            skipped_list = sorted(unmatched_prefixes)
            preview = ", ".join(skipped_list[:10])
            remaining = len(skipped_list) - 10
            if remaining > 0:
                preview += f", ... (+{remaining} more)"
            print(
                "Warning: the following per-layer bit overrides did not match any "
                f"quantizable layers and were ignored: {preview}"
            )

    loader = build_calibration_loader(model, config.calibration)
    activation_ranges = calibrate_model(
        model,
        loader,
        targets,
        device=device,
        max_batches=config.calibration.max_batches,
    )

    for name, target in targets.items():
        original_weight = target.module.weight.detach().cpu()
        qtensor = quantize_tensor(
            original_weight,
            target.num_bits,
            per_channel=target.per_channel,
            channel_axis=target.channel_axis,
        )
        dequantized = apply_quantized_weight(target, qtensor)
        target.observed_error = quantization_error(
            original_weight.to(dequantized.dtype),
            dequantized.cpu(),
        )

    artifact_paths: Dict[str, str] = {}
    if output_path:
        artifact_paths["checkpoint"] = str(output_path.resolve())
    if package_output_path:
        artifact_paths["package"] = str(package_output_path.resolve())

    metadata = dict(raw_metadata)
    metadata.update(
        {
            "bits": config.bits,
            "per_channel": config.per_channel,
            "activation_ranges": activation_ranges,
        }
    )
    metadata["per_layer_bits"] = {
        name: target.num_bits for name, target in targets.items()
    }
    if per_layer_bits:
        metadata["per_layer_bits_requested"] = per_layer_bits
        if unmatched_prefixes:
            metadata["per_layer_bits_unmatched_prefixes"] = sorted(unmatched_prefixes)
        if applied_overrides:
            metadata["per_layer_bits_applied"] = applied_overrides
    if artifact_paths:
        metadata.setdefault("artifacts", {}).update(artifact_paths)

    package = serialize_targets(
        targets,
        model_config=config.config_name,
        checkpoint_path=config.checkpoint_path,
        metadata=metadata,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, output_path)
    if package_output_path:
        package_output_path.parent.mkdir(parents=True, exist_ok=True)
        save_package(package, package_output_path)

    return package, targets, activation_ranges
