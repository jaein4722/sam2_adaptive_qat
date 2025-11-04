# Layer-wise Quantization for SAM 2

This project brings the core ideas of the CDPQ repository into the SAM 2
codebase without touching upstream modules. It focuses on post-training
quantization (PTQ) and a simplified confidence-driven progressive
quantization (CDPQ) workflow tailored to the SA-V dataset layout.

## Highlights
- Works entirely inside `projects/layerwise_quantization`—no patches to `sam2/` are required.
- Reuses the SA-V frame dataset (`../datasets/sa-v`) for calibration and evaluation.
- Provides weight-only quantization with optional per-channel scaling and layer-wise bit selection.
- Exports compact checkpoints that can be dequantized back into vanilla SAM 2 `state_dict` files when needed.

## Directory layout
| Path | Purpose |
| --- | --- |
| `data/` | Lightweight SA-V frame dataset loader for calibration. |
| `quantization/` | Quantizers, observers, and layer discovery helpers. |
| `cdpq/` | Feature capture, distillation losses, and confidence heuristics. |
| `adapters/` | SAM 2 model loader utilities. |
| `workflows/` | Calibration, PTQ, and CDPQ orchestration. |
| `io/` | Serialization helpers for quantized checkpoints. |
| `scripts/` | Ready-to-run commands (`train_cdpq.py`, `run_ptq.py`, `run_cdpq.py`, `progressive.sh`). |

## Quick start
1. Ensure the SA-V dataset is available at `../datasets/sa-v` (default from the
   repository instructions).
2. Launch CDPQ training with distillation:
   ```bash
   python -m projects.layerwise_quantization.scripts.train_cdpq \
       --teacher-config sam2.1/sam2.1_hiera_l \
       --teacher-checkpoint checkpoints/sam2.1_hiera_large.pt \
       --student-config sam2.1/sam2.1_hiera_b+ \
       --student-checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
       --train-root ../datasets/sa-v \
       --val-root ../datasets/sa-v \
       --epochs 5 \
       --batch-size 2
   ```
   The script logs to Weights & Biases, periodically evaluates distillation loss,
   performs confidence-driven bit reduction, and saves both float and quantized
   checkpoints under `sam2_logs/layerwise_quantization/`.
3. (Optional) Run standalone PTQ / CDPQ sweeps for ablations using `run_ptq.py`
   or `run_cdpq.py`. These commands reuse the calibration-only pipeline introduced
   earlier and offer a baseline for post-training-only experiments.

## Loading a quantized package into vanilla SAM 2
The quantized package stores integer weights plus scale/zero-point metadata.
To run inference with the unmodified `sam2.build_sam2` loader, convert the
package back into a floating `state_dict` before loading:

```python
import torch
from projects.layerwise_quantization.io.storage import load_package

package = load_package("sam2_logs/layerwise_quantization/student_quantized_package.pt")
state_dict = {}
for name, layer_state in package.layers.items():
    state_dict[name + ".weight"] = layer_state.tensor.dequantize()
quantized_metadata = package.metadata  # optional bookkeeping

# Combine with the rest of the checkpoint structure expected by SAM 2
checkpoint = {"model": state_dict}
torch.save(checkpoint, "sam2_logs/layerwise_quantization/student_quantized_float.pt")
```

You can now call `build_sam2(..., ckpt_path="sam2_logs/layerwise_quantization/student_quantized_float.pt")` without
modifying any upstream code.

## Configuration tips
- Use `--sample-count` and `--max-batches` to control calibration cost.
- Add `--skip-prefix encoder.backbone` (for example) to leave sensitive modules in
  full precision.
- Use `--target-roots image_encoder,memory_attention` to restrict bit reduction to
  specific top-level modules.
- CDPQ lowers bits sequentially; tighten `--confidence-threshold` to keep higher
  precision or loosen it to explore more aggressive compression.
- All metadata (activation ranges, chosen bit-widths, notes) is stored inside the
  quantized package for later analysis.
