"""CLI for running post-training quantization on SAM 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..workflows.calibration import CalibrationConfig
from ..workflows.ptq import PTQConfig, run_ptq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-wise PTQ for SAM 2")
    parser.add_argument("--config-name", required=True, help="Hydra config name for the SAM 2 model")
    parser.add_argument("--checkpoint", required=False, default=None, help="Path to model checkpoint")
    parser.add_argument("--dataset-root", default=None, help="Path to SA-V dataset root (defaults to ../datasets/sa-v)")
    parser.add_argument("--split", default="train", help="Dataset split: train, val, or test")
    parser.add_argument("--sample-count", type=int, default=512, help="Number of frames used for calibration")
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit calibration iterations")
    parser.add_argument("--bits", type=int, default=8, help="Target bit-width for quantization")
    parser.add_argument("--per-tensor", action="store_true", help="Use per-tensor quantization instead of per-channel")
    parser.add_argument("--skip-prefix", action="append", default=None, help="Module name prefix to skip (repeatable)")
    parser.add_argument("--device", default=None, help="Computation device (cuda, cpu, or auto)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the dequantized checkpoint compatible with sam2 loaders (.pt)",
    )
    parser.add_argument(
        "--package-output",
        default=None,
        help="Optional path for the serialized quantization package (defaults to <output>_quantized_package.pt when --output is set)",
    )
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

    config = PTQConfig(
        config_name=parsed.config_name,
        checkpoint_path=parsed.checkpoint,
        bits=parsed.bits,
        per_channel=not parsed.per_tensor,
        device=parsed.device,
        calibration=calibration,
        skip_prefixes=parsed.skip_prefix or (),
        output_path=parsed.output,
        package_output_path=parsed.package_output,
        metadata=extra_metadata,
    )

    package, targets, activation_ranges = run_ptq(config)

    artifacts = package.metadata.get("artifacts", {})
    checkpoint_path = artifacts.get("checkpoint")
    package_path = artifacts.get("package")

    if checkpoint_path:
        print(f"Dequantized checkpoint saved to {checkpoint_path}")
    elif parsed.output:
        print(f"Dequantized checkpoint saved to {Path(parsed.output).resolve()}")

    if package_path:
        print(f"Quantized package saved to {package_path}")
    elif parsed.package_output:
        print(f"Quantized package saved to {Path(parsed.package_output).resolve()}")

    print("Quantized layers:")
    for name, target in sorted(targets.items()):
        print(f"  {name}: {target.num_bits}-bit error={target.observed_error:.5f}")
    print(f"Activation ranges collected for {len(activation_ranges)} layers")


if __name__ == "__main__":
    main()
