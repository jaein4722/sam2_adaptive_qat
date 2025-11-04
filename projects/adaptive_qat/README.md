# Adaptive QAT Distillation

This project implements the teacher–student quantization workflow for SAM2 without modifying the core repository. The student model learns per-layer bitwidths under a knowledge distillation regime while sharing the original training losses.

## Components
- `models/system.py` builds teacher/student pairs, loads checkpoints, attaches fake-quant hooks, and exposes layer activations for distillation.
- `losses/adaptive_qat_loss.py` wraps the standard SAM2 segmentation loss and adds layer-wise KD, bit penalties, and output KD terms.
- `utils/` includes JSON importance loading, module resolution, activation quantizers, and feature capture helpers.
- `configs/adaptive_qat_template.yaml` mirrors the SAM2 training style and can be used as a starting point for Hydra launches.
- `train_adaptive_qat.py` is a thin wrapper around the existing `training.train` entry point.

## Workflow
1. Prepare datasets under `../datasets/sa-v` and `../datasets/sa-1b` (adjust paths in the config if needed).
2. Optionally edit `paths.importance_json` to point to a custom JSON with recommended bitwidths/importance values (same schema as `projects/layerwise_quantization/configs/ptq/bit_recommendation_v4_aggressive.json`).
3. Launch training:
   ```bash
   python projects/adaptive_qat/train_adaptive_qat.py --config projects/adaptive_qat/configs/adaptive_qat_template.yaml --experiment-dir <exp_dir>
   ```
4. Standard SAM2 inference scripts (`tools/vos_inference.py`, etc.) remain unchanged and continue to work with checkpoints produced by this project.
