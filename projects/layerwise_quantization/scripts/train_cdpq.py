"""CLI entry for CDPQ training on SAM 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..workflows.cdpq_training import (
    DataConfig,
    DistillConfig,
    ModelBuildConfig,
    QuantConfig,
    TrainingConfig,
    train_cdpq,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CDPQ on SAM 2")
    parser.add_argument("--teacher-config", required=True, help="Hydra config for teacher model")
    parser.add_argument("--teacher-checkpoint", required=True, help="Checkpoint for teacher model")
    parser.add_argument("--student-config", required=True, help="Hydra config for student model")
    parser.add_argument("--student-checkpoint", default=None, help="Checkpoint for student model (optional)")
    parser.add_argument("--train-root", default=None, help="Training dataset root")
    parser.add_argument("--val-root", default=None, help="Validation dataset root")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--initial-bits", type=int, default=8)
    parser.add_argument("--min-bits", type=int, default=4)
    parser.add_argument("--layer-threshold", type=float, default=0.1)
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--impact-threshold", type=float, default=0.3)
    parser.add_argument("--confidence-method", default="baseline", choices=["baseline", "variance"])
    parser.add_argument("--feature-limit", type=int, default=8)
    parser.add_argument(
        "--target-roots",
        default=None,
        help="Comma-separated module prefixes to restrict bit reduction (e.g., image_encoder,memory_attention)",
    )
    parser.add_argument("--wandb-project", default="CDPQ_SAM2")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tags", default=None, help="Comma separated tags")
    parser.add_argument("--save-dir", default="sam2_logs/layerwise_quantization")
    parser.add_argument("--train-sample-count", type=int, default=None)
    parser.add_argument("--val-sample-count", type=int, default=128)
    parser.add_argument("--train-scan", default=None, help="JSON string for DatasetScanConfig overrides")
    parser.add_argument("--val-scan", default=None, help="JSON string for DatasetScanConfig overrides")
    return parser


def parse_scan(json_str: str | None):
    from ..data.sav_dataset import DatasetScanConfig

    if not json_str:
        return DatasetScanConfig()
    payload = json.loads(json_str)
    return DatasetScanConfig(**payload)


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)

    teacher = ModelBuildConfig(
        config_name=parsed.teacher_config,
        checkpoint=parsed.teacher_checkpoint,
        device="auto",
    )
    student = ModelBuildConfig(
        config_name=parsed.student_config,
        checkpoint=parsed.student_checkpoint,
        device="auto",
    )

    train_data = DataConfig(
        root=parsed.train_root,
        split="sav_train",
        batch_size=parsed.batch_size,
        sample_count=parsed.train_sample_count,
        scan=parse_scan(parsed.train_scan),
    )
    val_data = DataConfig(
        root=parsed.val_root or parsed.train_root,
        split="sav_val",
        batch_size=parsed.val_batch_size,
        sample_count=parsed.val_sample_count,
        shuffle=False,
        scan=parse_scan(parsed.val_scan),
    )

    quant = QuantConfig(
        initial_bits=parsed.initial_bits,
        min_bits=parsed.min_bits,
    )

    target_roots = ()
    if parsed.target_roots:
        target_roots = tuple(filter(None, (root.strip() for root in parsed.target_roots.split(","))))

    distill = DistillConfig(
        feature_modules=None,
        feature_limit=parsed.feature_limit,
        loss_type="layer",
        layer_threshold=parsed.layer_threshold,
        variance_threshold=parsed.variance_threshold,
        impact_threshold=parsed.impact_threshold,
        confidence_method=parsed.confidence_method,
        target_roots=target_roots,
    )

    tags = tuple(filter(None, (parsed.tags.split(",") if parsed.tags else [])))

    config = TrainingConfig(
        teacher=teacher,
        student=student,
        train_data=train_data,
        val_data=val_data,
        quant=quant,
        distill=distill,
        epochs=parsed.epochs,
        learning_rate=parsed.learning_rate,
        project=parsed.wandb_project,
        entity=parsed.wandb_entity,
        run_name=parsed.run_name,
        tags=tags,
        save_dir=parsed.save_dir,
    )

    results = train_cdpq(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
