"""SA-V frame dataset utilities for quantization calibration.

The dataset loader intentionally avoids dependencies on the training
infrastructure under ``projects/sav_finetune`` to keep this package isolated.
It scans the standard SA-V directory layout and exposes a lightweight
``torch.utils.data.Dataset`` that yields RGB frames as ``PIL.Image`` objects
or tensors after applying an optional transform.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset

__all__ = ["DEFAULT_ROOT", "SAVFrameDataset", "build_frame_dataset"]

DEFAULT_ROOT = Path("../datasets/sa-v").resolve()

@dataclass(frozen=True)
class DatasetScanConfig:
    """Configuration used when scanning SA-V frame directories."""

    split: str = "sav_train"
    frame_dirname: str = "JPEGImages_24fps"
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png")
    max_videos: Optional[int] = None
    max_frames_per_video: Optional[int] = None


class SAVFrameDataset(Dataset):
    """Lightweight frame dataset for SA-V calibration and evaluation."""

    def __init__(
        self,
        root: Optional[Path | str] = None,
        split: str = "train",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        sample_count: Optional[int] = None,
        scan_config: Optional[DatasetScanConfig] = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        if not self.root.exists():
            raise FileNotFoundError(f"SA-V dataset root not found: {self.root}")

        normalized_split = self._normalize_split(split)
        self.split = normalized_split
        self.transform = transform
        self.sample_count = sample_count
        self.scan_config = scan_config or DatasetScanConfig(split=normalized_split)

        self._rng = random.Random(seed)
        self._frame_paths = self._scan_frames()
        if not self._frame_paths:
            raise RuntimeError(
                "No frames discovered for split "
                f"{normalized_split!r} under {self.root}"
            )

        if sample_count is not None and sample_count < len(self._frame_paths):
            self._frame_paths = self._rng.sample(self._frame_paths, sample_count)

    @staticmethod
    def _normalize_split(split: str) -> str:
        split = split.lower()
        if split in {"train", "sav_train"}:
            return "sav_train"
        if split in {"val", "valid", "validation", "sav_val"}:
            return "sav_val"
        if split in {"test", "eval", "sav_test"}:
            return "sav_test"
        raise ValueError(f"Unsupported SA-V split: {split}")
    def _scan_frames(self) -> List[Path]:
        cfg = self.scan_config
        split_root = self.root / cfg.split
        frame_root = split_root / cfg.frame_dirname
        if not frame_root.exists():
            raise FileNotFoundError(
                f"Expected frame directory {frame_root} does not exist"
            )

        video_dirs = sorted(p for p in frame_root.iterdir() if p.is_dir())
        if cfg.max_videos is not None:
            video_dirs = video_dirs[: cfg.max_videos]

        collected: List[Path] = []
        for video_dir in video_dirs:
            frame_files = sorted(
                p
                for p in video_dir.iterdir()
                if p.suffix.lower() in cfg.extensions and p.is_file()
            )
            if cfg.max_frames_per_video is not None:
                frame_files = frame_files[: cfg.max_frames_per_video]
            collected.extend(frame_files)
        return collected
    def __len__(self) -> int:
        return len(self._frame_paths)

    def __getitem__(self, index: int) -> dict:
        path = self._frame_paths[index]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image) if self.transform else image
        return {"image": tensor, "path": str(path)}

    def iter_paths(self) -> Iterable[Path]:
        yield from self._frame_paths


def build_frame_dataset(
    *,
    root: Optional[Path | str] = None,
    split: str = "train",
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    sample_count: Optional[int] = None,
    scan_config: Optional[DatasetScanConfig] = None,
    seed: int = 42,
) -> SAVFrameDataset:
    """Factory used by workflows to build a calibration dataset."""

    return SAVFrameDataset(
        root=root,
        split=split,
        transform=transform,
        sample_count=sample_count,
        scan_config=scan_config,
        seed=seed,
    )
