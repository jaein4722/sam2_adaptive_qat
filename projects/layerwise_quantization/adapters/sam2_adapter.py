"""Convenience helpers for instantiating SAM 2 models inside quantization workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch

from sam2.build_sam import build_sam2

__all__ = [
    "build_sam2_model",
    "ensure_device",
    "named_quantizable_layers",
]


def ensure_device(device: Optional[str | torch.device]) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _normalize_config_name(config_name: str) -> str:
    text = config_name
    if text.endswith(".yaml"):
        return text if text.startswith("configs/") else f"configs/{text}"
    if text.startswith("configs/"):
        return f"{text}.yaml"
    return f"configs/{text}.yaml"


def build_sam2_model(
    config_name: str,
    checkpoint_path: Optional[str] = None,
    *,
    device: Optional[str | torch.device] = None,
    hydra_overrides: Optional[Sequence[str]] = None,
    mode: str = "eval",
) -> torch.nn.Module:
    normalized_config = _normalize_config_name(config_name)
    model = build_sam2(
        config_file=normalized_config,
        ckpt_path=checkpoint_path,
        device=str(ensure_device(device)),
        mode=mode,
        hydra_overrides_extra=list(hydra_overrides or []),
    )
    return model


def named_quantizable_layers(
    model: torch.nn.Module,
    allowed_types: Sequence[type],
) -> List[tuple[str, torch.nn.Module]]:
    layers: List[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, tuple(allowed_types)) and hasattr(module, "weight"):
            layers.append((name, module))
    return layers
