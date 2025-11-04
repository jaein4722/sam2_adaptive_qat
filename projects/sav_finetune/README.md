# SA-V Fine-tuning Utilities

This folder contains a small workflow for fine-tuning SAM 2 on the SA-V dataset and evaluating the resulting checkpoint on SA-V val/test splits. It reuses the official training stack while keeping the configuration and helper scripts in one place.

## 1. Prepare the dataset
1. Extract the SA-V training videos to JPEG frames using `training/scripts/sav_frame_extraction_submitit.py` or your preferred extractor. After this step each video should live under `JPEGImages_24fps/{video_id}/00000.jpg`, etc.
2. Keep the released `*_manual.json` files together in a directory (e.g. `SA-V/train/annotations`). Pass this directory to `--train-annotations` when launching training.
3. For evaluation, download the SA-V val/test packages so that you have folders such as:
   - `<SPLIT>/JPEGImages_24fps/{video_id}/` (RGB frames at 24 fps)
   - `<SPLIT>/Annotations_6fps/{video_id}/{object_id}/frame.png` (per-object masks)
   - Optional `sav_val.txt` / `sav_test.txt` files listing the video ids.

## 2. Fine-tune SAM 2 on SA-V
`train_sav.py` loads the Hydra configuration from `configs/sam2.1_hiera_b+_sav_finetune.yaml`, applies a few convenient CLI overrides, and then reuses `training/train.py`'s launch utilities. Example (single node, 8 GPUs):

```bash
python projects/sav_finetune/train_sav.py \
  --train-frames /path/to/sav_train/JPEGImages_24fps \
  --train-annotations /path/to/sav_train/annotations \
  --file-list /path/to/sav_train.txt \  # optional subset
  --checkpoint /path/to/sam2.1_hiera_base_plus.pt \
  --experiment-dir /path/to/exp/sav_finetune \   # optional, otherwise sam2_logs/<config_name>
  --gpus 8
```

Notable flags:
- `--batch-size`, `--epochs`, `--num-workers`, `--seed` override the defaults in the config.
- `--resume` lets you continue from an intermediate checkpoint.
- `--use-cluster 1` plus SLURM flags will submit to a cluster via Submitit with the same options as the standard launcher.

The full Hydra config lives in `configs/sam2.1_hiera_b+_sav_finetune.yaml`; edit it directly if deeper changes are required (e.g. learning rates, augments).

## 3. Run VOS inference and compute SA-V metrics
`evaluate_sav.py` wraps `tools/vos_inference.py` and `sav_dataset/sav_evaluator.py`. It generates predictions in the SA-V directory layout and immediately calls the official evaluator.

```bash
python projects/sav_finetune/evaluate_sav.py \
  --checkpoint /path/to/exp/sav_finetune/checkpoints/last.ckpt \
  --video-root /path/to/sav_val/JPEGImages_24fps \
  --annotation-root /path/to/sav_val/Annotations_6fps \
  --video-list /path/to/sav_val.txt \        # optional
  --output-root /tmp/sav_val_predictions \
  --metrics-json /tmp/sav_val_metrics.json
```

Key options:
- `--model-cfg` controls which SAM 2 predictor backbone to instantiate (defaults to the base-plus config).
- `--use-all-masks` will feed annotated frames beyond the first to SAM 2 (by default only frame 0 is used, matching the paper setup).
- `--skip-inference` skips mask generation and only evaluates an existing prediction directory.
- `--do-not-skip-first-last` disables the standard SA-V practice of dropping the first/last annotated frame when computing J/F scores.

The script prints the aggregated J&F / J / F values and (optionally) stores them as JSON alongside per-sequence breakdowns. A CSV with the same breakdown is also written by the evaluator inside the predictions folder.

## 4. Notes
- Install the training extras once (`pip install -e ".[dev]"`) before running the scripts.
- Make sure your working directory is the repository root so the Hydra configs under `sam2/` remain discoverable.
- The utilities do not alter existing training code; they only orchestrate the workflow and keep SA-V-specific paths/configuration separate from other experiments.
