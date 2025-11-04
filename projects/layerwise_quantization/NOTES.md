# Layerwise Quantization – Development Notes

_Last updated: 2025-09-29_

## 1. Current Implementation Overview

### 1.1 Project Layout
```
projects/layerwise_quantization/
├── adapters/          # SAM 2 model loaders and wrappers
├── cdpq/              # Feature capture, distillation losses, confidence heuristics
├── data/              # SA-V frame dataset helper
├── io/                # Quantized package serialization
├── quantization/      # Weight quantization wrappers/utilities
├── scripts/           # CLI entry points & progressive.sh helper
└── workflows/         # PTQ + CDPQ orchestration
```

### 1.2 CDPQ Training Pipeline (`workflows/cdpq_training.py`)
- Loads teacher/student SAM 2 models via `adapters.sam2_adapter.build_sam2_model`.
- Wraps student Conv/Linear layers with `QuantizedModuleMixin` (weight-only quantization, dequantize on fly).
- Distillation loss (`cdpq.distillation.compute_distillation_loss`):
  - Default: layer statistics (mean/variance) MSE.
  - Optional affinity loss.
- Training loop:
  - Distillation loss only; optional gradient clipping.
  - `tqdm` progress + structured logging to console & `train.log`.
  - Each run stored in timestamped directory under `sam2_logs/layerwise_quantization/<timestamp>_<run_name>` with `config.json`, `train.log`, float checkpoint, quantized package.
  - wandb logging with offline fallback.
- Validation & bit reduction:
  - Collect per-layer distillation loss history.
  - Confidence heuristics: baseline (mean) or variance (mean + var thresholds).
  - Optional `--target-roots` to constrain bit reduction to top-level modules.
  - `_apply_bit_reduction` evaluates loss after decrementing bit width; accepts if loss increase ≤ `impact_threshold`.
  - Logs candidate count, per-layer attempt (loss, increase, status), accept rate, mean increase.
- Final artifacts:
  - Float checkpoint (`student_quantized_float.pt`).
  - Quantized package (`student_quantized_package.pt`) via `io.serialize_targets` (stores int weights + scales/zero-points).
  - Results summary printed and logged.

### 1.3 CLI (`scripts/train_cdpq.py`)
- Provides flags for teacher/student configs, dataset paths, epochs, LR, initial/min bits, confidence method, target roots, etc.
- Parses optional `DatasetScanConfig` overrides (JSON).
- Populates `TrainingConfig` dataclass and launches workflow.
- Additional ready-to-run shell script: `scripts/progressive.sh` (mirrors `train_sav.sh` style).

### 1.4 PTQ utilities
- `scripts/run_ptq.py`, `workflows/ptq.py`: baseline post-training quantization using calibration dataset.
- Activation ranges recorded, quantized package saved similarly.

### 1.5 Quantization Modules (`quantization/modules.py`)
- `QuantizedConv2d` / `QuantizedLinear` wrap original modules.
- `quantize_weight()` quantizes weight each forward pass (per-channel symmetric) and dequantizes immediately for FP32 computation.
- Observed error tracked for reporting; actual compression happens when saving via `quantized_package`.

### 1.6 CDPQ utilities (`cdpq/`)
- `features.py`: Hook-based feature capture (with/without gradient detachment).
- `distillation.py`: implements layer-statistics and affinity loss.
- `confidence.py`: baseline & variance heuristics for candidate selection.

### 1.7 Logging & Experiment Management
- Train/val progress printed to console with epoch summaries.
- `train.log` mirrors console output; `config.json` stores args.
- WandB logging remains optional (automatic offline fallback).
- Quantization attempts and acceptance stats logged with details.


## 2. Key Design Choices

1. **Weight-only quantization** during training (activations remain FP32).
2. **Distillation-only loss** for both training and evaluation.
3. **Confidence-based bit reduction** (mean/variance heuristics) combining history with candidate filtering.
4. **Per-layer evaluation during bit reduction** to ensure loss increase under control (accept/reject + rollback).
5. **Timestamped experiment directories** to avoid overwriting results.
6. **Separation of concerns**: adapters, CDPQ utilities, quantization core, workflows, scripts.


## 3. Outstanding Ideas & Future Directions

1. **Integrate Quantization into Objective**
   - Move from heuristic bit reduction to a joint loss (task + distillation + bit penalty).
   - Explore continuous relaxation of bit-width or explicit regularization terms.

2. **Activation Quantization**
   - Introduce fake quant / observer modules for activations.
   - Manage clipping, per-layer sensitivity, transformer-specific concerns.

3. **Q/K/V Separation**
   - Inspired by SlimSAM: split QKV projections for finer control over quantization/pruning.
   - Evaluate efficiency trade-offs and compatibility with existing checkpoints.

4. **Additional Confidence Heuristics**
   - Extend beyond mean/variance: trend analysis, stability windows, dynamic thresholds.
   - Use distillation gradients or attention-specific metrics.

5. **Mixed Precision & Continuous Penalty**
   - Introduce penalty terms tied to model size/FLOPs to guide bit reduction automatically.
   - Possibly unify with objective-based quantization to eliminate separate confidence stage.

6. **Activation Distillation Penalties**
   - Layer-level scales for activation distillation to align with future activation quant.

7. **Compression Evaluation**
   - Add tooling for measuring actual model size, runtime, accuracy after quantized package reloading.

8. **Loss/Logging Enhancements**
   - Distinguish contributions by module (e.g., image encoder vs. decoder) for detailed analysis.


## 4. Open Questions

- How to systematically design penalty coefficients when integrating bit width into the loss?
- What is the most stable strategy for activation quantization in SAM2 (given attention sensitivity)?
- How aggressive can q/k/v separation be without harming throughput? (Need benchmarks.)
- How to generalize `target_roots` filtering for nested modules or dynamic attention blocks?


## 5. References & Related Work (for future study)

- Quantization-aware training: DoReFa-Net, PACT, LSQ, QAT on Transformers (Q-BERT, etc.).
- Knowledge distillation for quantization: PKD, QKD, etc.
- Mixed-precision exploration: HAQ, OCS, Grace, etc.
- SlimSAM for qkv structural modifications.


---

_These notes are intended for internal reference while evolving the CDPQ/quantization pipeline for SAM 2. They summarize the current implementation, highlight design choices, and outline potential future work._
