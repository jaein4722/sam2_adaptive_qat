# Layerwise QAT Distillation Quantization Plan

## 1. Objectives & Scope
- Implement student-teacher QAT workflow with per-layer adaptive bitwidths.
- Maintain shared architecture; only precision differs between teacher and student.
- Support layerwise KD, output KD, CE loss, and bit-penalty integration inside the standard training loop.
- Provide logging, config hooks, and safety checks for dynamic bit adaptation.

## 2. Model Preparation
### 2.1 Teacher Setup
- Load pretrained teacher checkpoint; freeze parameters; ensure eval-mode forward without fake quant.
- Expose forward hooks or interfaces to capture intermediate activations matching student layer order.

### 2.2 Student Wrapper
- Extend base SAM2 module with QAT-ready wrappers that insert fake quant modules at layer outputs (and weights if needed).
- Introduce `BitController` holding float bit variables `b_i` aligned with each quantized layer.
- Ensure fake quant modules accept on-the-fly integer bitwidth (cast from `b_i`) while retaining float storage for gradients.
- Add utilities to sample layer activations for KD (e.g., register forward hooks emitting tensors before fake quant output).
### 2.3 Bit Variable Management
- Define dataclass/config for layer importance values, min/max bit limits, optional temperature/annealing parameters.
- Provide mapping from module names to bit indices; expose API to query/update current bitwidth snapshot.
- Implement gradient-safe clamping (e.g., `b_i = clamp(b_i, min_bits, max_bits)`) post-update without breaking float gradients.

## 3. Loss Computation Graph
### 3.1 Layerwise KD Losses
- During forward, cache teacher/student activations per target layer (`L_kd_i` computation point).
- Support multiple distance metrics (MSE default; optional KL for logits) with configurable weighting.
- Accumulate `L_kd_i` tensors and reuse them when forming `L_bit_i` to avoid redundant computation.

### 3.2 Bit Penalty
- Compute `L_bit_i = L_kd_i + (b_i / importance_i)` with broadcasting safety and optional importance normalization.
- Allow global scalar to scale penalty vs. KD component; expose via config.
- Log raw `b_i`, importance, and resulting penalty for downstream analysis.

### 3.3 Output Losses
- Compute `L_CE` between student logits and ground-truth labels/masks.
- Compute `L_KD_output` between teacher and student final outputs (e.g., soft targets with temperature scaling).
- Support optional mixing coefficient (`lambda_output`) to balance KD vs CE.
## 4. Training Loop Integration
- Extend existing trainer (e.g., `training/trainer.py` or project-specific loop) with teacher forward pass and loss aggregation.
- Register optimizer param groups: weights, biases, bit variables (optionally distinct LR/regularization).
- Insert backward/step order: zero grads → forward both models → compute composite losses → backward → clip bit gradients if needed → optimizer step → clamp bit vars → update schedulers.
- Update logging hooks to record losses, bit stats, and optional histograms.

## 5. Fake Quant Modules
- Implement custom fake quant function/class that consumes float `b_i`, casts to int during quantization, and supports autograd.
- For weight quantization, wrap modules (e.g., conv/linear) with quantized weight proxies; ensure inference compatibility.
- For activation quantization, optionally insert observer modules to track ranges; reuse if PyTorch QAT API is available.
- Provide toggles to disable quantization during evaluation or ablations.

## 6. Importance Handling & Hyperparameters
- Load importance values from config or external CSV/JSON; validate alignment with layer ordering.
- Offer strategies to derive importance (static file, gradient-based estimates, uniform fallback).
- Add CLI/Hydra overrides for: initial bit values, min/max bits, penalty scaling, KD weights, importance file path.
## 7. Logging & Diagnostics
- Integrate with existing logging framework (wandb/tensorboard/csv) to snapshot bitwidths and layer penalties per step.
- Emit periodic summaries (min/max/mean bits, loss components) and detect anomalies (e.g., bits stuck at bounds).
- Store checkpoints with metadata capturing current bit variables for reproducibility.

## 8. Evaluation Pipeline
- During eval, freeze bit variables → cast to nearest int → run student inference with fake quant disabled or set to fixed bits.
- Compare against teacher/full-precision baselines; track accuracy vs. compression metrics.
- Provide scripts to sweep bit penalty weights and record trade-off curves.

## 9. Testing Strategy
- Unit-test bit controller (clamping, casting, gradient flow) using synthetic modules under `tests/projects/layerwise_quantization/`.
- Add regression test for loss assembly ensuring `L_total` matches spec given mocked tensors.
- Include integration smoke test: single batch forward/backward using dummy data to confirm no runtime errors.

## 10. Development Milestones
- M1: Scaffold student wrapper, fake quant, bit controller with unit tests.
- M2: Integrate layerwise KD capture and loss computation, verify on toy run.
- M3: Hook into training loop, log bit dynamics, tune hyperparameters.
- M4: Full experiment sweep with evaluation scripts and documentation update.
