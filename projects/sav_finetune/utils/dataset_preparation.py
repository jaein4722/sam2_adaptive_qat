"""Utilities to prepare SA-V datasets for SAM 2 training."""

import logging
from pathlib import Path
from typing import Iterable, Optional


def _has_extracted_frames(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(child.is_dir() and any(grandchild.suffix.lower() in {".jpg", ".jpeg", ".png"} for grandchild in child.iterdir()) for child in path.iterdir())


def _iter_videos(root: Path) -> Iterable[Path]:
    for ext in (".mp4", ".mov", ".mkv"):
        yield from root.rglob(f"*{ext}")


def _extract_video(video_path: Path, output_dir: Path) -> None:
    from torchvision.io import VideoReader  # lazy import to avoid global dependency
    from PIL import Image

    try:
        reader = VideoReader(str(video_path), "video")
    except RuntimeError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Failed to open video {}. Torchvision might be missing FFmpeg support.".format(video_path)
        ) from exc

    reader.set_current_stream("video")

    output_dir.mkdir(parents=True, exist_ok=True)

    sentinel = output_dir / ".extracted"
    if sentinel.exists():
        # Already extracted
        return

    index = 0
    for record in reader:
        frame = record["data"].cpu().numpy()  # (H, W, C), uint8
        img = Image.fromarray(frame)
        img.save(output_dir / f"{index:05d}.jpg")
        index += 1

    sentinel.write_text("ok")


def prepare_training_frames_if_needed(
    video_root: str,
    target_root: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[str]:
    """Ensure training videos exist as frame sequences on disk.

    Args:
        video_root: Directory that might contain MP4 videos.
        target_root: Directory to place the extracted frames. Defaults to
            ``<video_root>/JPEGImages_auto``.
        overwrite: If True, re-extract frames even if they exist.

    Returns:
        The path that should be used as ``cfg.dataset.img_folder``. ``None``
        is returned when no videos were found and no action was required.
    """
    input_path = Path(video_root)
    if not input_path.exists() or not input_path.is_dir():
        logging.warning("Video root %s does not exist or is not a directory", video_root)
        return None

    videos = [path for path in _iter_videos(input_path)]
    if not videos:
        # Nothing to do – we assume the directory already contains frames.
        return None

    frames_root = Path(target_root) if target_root else input_path / "JPEGImages_auto"
    frames_root.mkdir(parents=True, exist_ok=True)

    has_frames = _has_extracted_frames(frames_root)
    if has_frames and not overwrite:
        return str(frames_root)

    logging.info("Extracting %d training videos into %s", len(videos), frames_root)

    error_paths = []
    for idx, video_path in enumerate(sorted(videos), start=1):
        video_name = video_path.stem
        dest_dir = frames_root / video_name
        try:
            if dest_dir.exists() and not overwrite and any(dest_dir.glob("*.jpg")):
                continue
            logging.debug("[%d/%d] Extracting %s", idx, len(videos), video_path)
            _extract_video(video_path, dest_dir)
        except Exception as exc:  # pragma: no cover - defensive programming
            logging.error("Failed to extract %s: %s", video_path, exc)
            error_paths.append(video_path)

    if error_paths:
        raise RuntimeError(
            "Failed to extract frames for the following videos: "
            + ", ".join(str(path) for path in error_paths)
        )

    return str(frames_root)
