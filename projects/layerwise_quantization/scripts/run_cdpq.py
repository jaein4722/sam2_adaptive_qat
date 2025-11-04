"""CLI for confidence-driven progressive quantization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..workflows.calibration import CalibrationConfig
from ..workflows.cdpq import CDPQConfig, run_cdpq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Confidence-driven progressive quantization for SAM 2")
    parser.add_argument("--config-name", required=True, help="Hydra config name for the SAM 2 model")
    parser.add_argument("--checkpoint", required=False, default=None, help="Path to model checkpoint")
    parser.add_argument("--dataset-root", default=None, help="Path to SA-V dataset root")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--sample-count", type=int, default=512, help="Number of frames used for calibration")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit calibration iterations")
    parser.add_argument("--bit-candidates", type=int, nargs="+", default=[8, 6, 4], help="Bit-widths to attempt (high to low)")
    parser.add_argument("--confidence-threshold", type=float, default=0.03, help="Maximum relative error tolerated when lowering bits")
    parser.add_argument("--min-bits", type=int, default=None, help="Minimum bit-width allowed")
    parser.add_argument("--max-bits", type=int, default=None, help="Maximum bit-width allowed")
    parser.add_argument("--per-tensor", action="store_true", help="Use per-tensor quantization")
    parser.add_argument("--skip-prefix", action="append", default=None, help="Module name prefix to skip")
    parser.add_argument("--device", default=None, help="Computation device")
    parser.add_argument("--output", default=None, help="Output path for the quantized package")
    parser.add_argument("--metadata", default=None, help="JSON string with extra metadata")
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)

    calibration = CalibrationConfig(
        dataset_root=parsed.dataset_root,
        split=parsed.split,
        sample_count=parsed.sample_count,
        batch_size=parsed.batch_size,
        num_workers=parsed.num_workers,
        max_batches=parsed.max_batches,
    )

    extra_metadata = {}
    if parsed.metadata:
        extra_metadata = json.loads(parsed.metadata)

    config = CDPQConfig(
        config_name=parsed.config_name,
        checkpoint_path=parsed.checkpoint,
        bits=max(parsed.bit_candidates),
        per_channel=not parsed.per_tensor,
        device=parsed.device,
        calibration=calibration,
        skip_prefixes=parsed.skip_prefix or (),
        output_path=parsed.output,
        metadata=extra_metadata,
        bit_candidates=parsed.bit_candidates,
        confidence_threshold=parsed.confidence_threshold,
        min_bits=parsed.min_bits,
        max_bits=parsed.max_bits,
    )

    package, targets, activation_ranges = run_cdpq(config)

    if parsed.output:
        print(f"Quantized package saved to {Path(parsed.output).resolve()}")

    print("Quantized layers:")
    for name, target in sorted(targets.items()):
        note = target.notes[-1] if target.notes else ""
        print(
            f"  {name}: {target.num_bits}-bit error={target.observed_error:.5f} {note}"
        )
    print(f"Activation ranges collected for {len(activation_ranges)} layers")


if __name__ == "__main__":
    main()
