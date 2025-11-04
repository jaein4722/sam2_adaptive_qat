"""Calibration utilities shared by PTQ and CDPQ workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from sam2.utils.transforms import SAM2Transforms

from ..data.sav_dataset import build_frame_dataset
from ..quantization.observers import collect_activation_ranges, release_observers
from ..quantization.utils import QuantizationTarget

__all__ = [
    "CalibrationConfig",
    "build_calibration_loader",
    "calibrate_model",
]

@dataclass
class CalibrationConfig:
    dataset_root: Optional[str] = None
    split: str = "train"
    sample_count: Optional[int] = 512
    batch_size: int = 1
    num_workers: int = 4
    max_batches: Optional[int] = None
    mask_threshold: float = 0.0
    max_hole_area: float = 0.0
    max_sprinkle_area: float = 0.0

def build_calibration_loader(
    model: torch.nn.Module,
    config: CalibrationConfig,
) -> DataLoader:
    transforms = SAM2Transforms(
        model.image_size,
        mask_threshold=config.mask_threshold,
        max_hole_area=config.max_hole_area,
        max_sprinkle_area=config.max_sprinkle_area,
    )
    dataset = build_frame_dataset(
        root=config.dataset_root,
        split=config.split,
        transform=transforms,
        sample_count=config.sample_count,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

@torch.no_grad()
def calibrate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    targets: Dict[str, QuantizationTarget],
    *,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    model.eval()
    model_device = device or next(model.parameters()).device
    modules = [(name, target.module) for name, target in targets.items()]
    observers, handles = collect_activation_ranges(model, modules)

    processed = 0
    try:
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch["image"]
            if isinstance(images, torch.Tensor):
                input_tensor = images.to(model_device, non_blocking=True)
            else:
                raise TypeError("Calibration loader must return tensors for 'image'.")
            _ = model.forward_image(input_tensor)
            processed += input_tensor.size(0)
    finally:
        release_observers(handles)

    activation_ranges = {}
    for name, observer in observers.items():
        try:
            activation_ranges[name] = observer.as_tuple()
        except RuntimeError:
            activation_ranges[name] = (0.0, 0.0)
    return activation_ranges
