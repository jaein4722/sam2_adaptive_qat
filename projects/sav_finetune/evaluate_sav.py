"""Inference and evaluation utilities for SA-V fine-tuning."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from typing import Dict, List, Optional, Sequence

import torch

from sam2.build_sam import build_sam2_video_predictor
from sav_dataset.utils.sav_benchmark import benchmark
from tools.vos_inference import vos_inference

DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def _discover_videos(video_root: Path, video_list_file: Optional[Path]) -> List[str]:
    if video_list_file is not None:
        return [v.strip() for v in video_list_file.read_text().splitlines() if v.strip()]
    return sorted([p.name for p in video_root.iterdir() if p.is_dir()])


def _run_prediction(
    predictor,
    video_root: Path,
    annotation_root: Path,
    output_root: Path,
    video_names: Sequence[str],
    score_thresh: float,
    use_all_masks: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for idx, video_name in enumerate(video_names, start=1):
        print(f"\n{idx}/{len(video_names)} - running inference on {video_name}")
        vos_inference(
            predictor=predictor,
            base_video_dir=str(video_root),
            input_mask_dir=str(annotation_root),
            output_mask_dir=str(output_root),
            video_name=video_name,
            score_thresh=score_thresh,
            use_all_masks=use_all_masks,
            per_obj_png_file=True,
        )


def _summarize_metrics(
    all_global_jf: Sequence[float],
    all_global_j: Sequence[float],
    all_global_f: Sequence[float],
    all_object_metrics: Sequence[Dict[str, Dict[str, float]]],
) -> Dict[str, object]:
    summary = {
        "global_JF": all_global_jf[0] if all_global_jf else None,
        "global_J": all_global_j[0] if all_global_j else None,
        "global_F": all_global_f[0] if all_global_f else None,
        "per_sequence": all_object_metrics[0] if all_object_metrics else {},
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM 2 inference on SA-V videos and compute evaluation metrics",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the fine-tuned SAM 2 checkpoint",
    )
    parser.add_argument(
        "--model-cfg",
        default=DEFAULT_MODEL_CFG,
        help="Model config name used to build the video predictor",
    )
    parser.add_argument(
        "--video-root",
        required=True,
        help="Directory with per-video folders of RGB frames (e.g. JPEGImages_24fps)",
    )
    parser.add_argument(
        "--annotation-root",
        required=True,
        help="Directory with ground-truth masks per object (e.g. Annotations_6fps)",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory to store predicted masks",
    )
    parser.add_argument(
        "--video-list",
        type=str,
        default=None,
        help="Optional text file with one video id per line",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
        help="Score threshold passed to vos_inference",
    )
    parser.add_argument(
        "--use-all-masks",
        action="store_true",
        help="Whether to use all available annotated frames as input",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used for inference",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only run evaluation using existing predictions in output_root",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default=None,
        help="Optional path to store evaluation metrics as JSON",
    )
    parser.add_argument(
        "--do-not-skip-first-last",
        action="store_true",
        help="Disable skipping the first and last annotated frames during evaluation",
    )

    args = parser.parse_args()

    video_root = Path(args.video_root)
    annotation_root = Path(args.annotation_root)
    output_root = Path(args.output_root)
    video_list_file = Path(args.video_list) if args.video_list else None

    video_names = _discover_videos(video_root, video_list_file)
    print(f"Running on {len(video_names)} videos")

    if not args.skip_inference:
        hydra_overrides_extra = ["++model.non_overlap_masks=false"]
        predictor = build_sam2_video_predictor(
            config_file=args.model_cfg,
            ckpt_path=args.checkpoint,
            device=args.device,
            apply_postprocessing=True,
            hydra_overrides_extra=hydra_overrides_extra,
        )
        torch.set_grad_enabled(False)
        _run_prediction(
            predictor,
            video_root,
            annotation_root,
            output_root,
            video_names,
            score_thresh=args.score_thresh,
            use_all_masks=args.use_all_masks,
        )
    else:
        print("Skipping inference and using existing predictions")

    skip_first_and_last = not args.do_not_skip_first_last
    all_global_jf, all_global_j, all_global_f, all_object_metrics = benchmark(
        [str(annotation_root)],
        [str(output_root)],
        strict=True,
        num_processes=None,
        verbose=True,
        skip_first_and_last=skip_first_and_last,
    )

    summary = _summarize_metrics(
        all_global_jf,
        all_global_j,
        all_global_f,
        all_object_metrics,
    )
    print("\nAggregated SA-V metrics:")
    print(json.dumps(summary, indent=2))

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
