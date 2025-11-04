"""CDPQ training workflow adapted for SAM 2."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import logging
import re
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:  # optional wandb fallback
    import wandb
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without wandb
    class _WandbStub:  # simple no-op logger
        def init(self, **kwargs):
            print("[WARN] wandb not installed. Logging disabled.")
            return self

        def log(self, *args, **kwargs):
            pass

        @property
        def run(self):
            return None

    wandb = _WandbStub()

from sam2.utils.transforms import SAM2Transforms

from ..data.sav_dataset import DatasetScanConfig, build_frame_dataset
from ..cdpq.distillation import compute_distillation_loss
from ..cdpq.features import capture_features
from ..quantization.utils import collect_quantized_layers, wrap_model_for_quantization
from ..adapters.sam2_adapter import build_sam2_model, ensure_device
from ..cdpq.confidence import confidence_baseline, confidence_variance

__all__ = ["ModelBuildConfig", "DataConfig", "QuantConfig", "DistillConfig", "TrainingConfig", "train_cdpq"]


@dataclass
class ModelBuildConfig:
    config_name: str
    checkpoint: Optional[str]
    device: str = "auto"
    hydra_overrides: Tuple[str, ...] = ()


@dataclass
class DataConfig:
    root: Optional[str] = None
    split: str = "sav_train"
    batch_size: int = 1
    num_workers: int = 0
    sample_count: Optional[int] = None
    shuffle: bool = True
    scan: DatasetScanConfig = field(default_factory=DatasetScanConfig)


@dataclass
class QuantConfig:
    initial_bits: int = 8
    min_bits: int = 4
    per_channel: bool = True
    skip_prefixes: Tuple[str, ...] = ()


@dataclass
class DistillConfig:
    feature_modules: Optional[List[str]] = None
    feature_limit: int = 8
    loss_type: str = "layer"
    layer_threshold: float = 0.1
    variance_threshold: float = 0.05
    impact_threshold: float = 0.3
    confidence_method: str = "baseline"  # baseline | variance
    target_roots: Tuple[str, ...] = ()


@dataclass
class TrainingConfig:
    teacher: ModelBuildConfig
    student: ModelBuildConfig
    train_data: DataConfig
    val_data: DataConfig
    quant: QuantConfig
    distill: DistillConfig
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: Optional[float] = 1.0
    log_interval: int = 10
    eval_interval: int = 1
    project: str = "CDPQ_SAM2"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Tuple[str, ...] = ()
    save_dir: str = "sam2_logs/layerwise_quantization"


def _init_wandb(cfg: TrainingConfig) -> None:
    try:
        wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=cfg.run_name,
            tags=list(cfg.tags),
            config={
                "epochs": cfg.epochs,
                "lr": cfg.learning_rate,
                "initial_bits": cfg.quant.initial_bits,
                "min_bits": cfg.quant.min_bits,
                "confidence_method": cfg.distill.confidence_method,
                "loss_type": cfg.distill.loss_type,
            },
            resume="allow",
        )
    except Exception as exc:  # pragma: no cover - runtime fallback
        print(f"[WARN] wandb init failed ({exc}); continuing without logging.")
        class _WandbNoOp:
            def log(self, *args, **kwargs):
                pass

            @property
            def run(self):
                return None

        globals()["wandb"] = _WandbNoOp()


def _build_dataloader(cfg: DataConfig, *, resolution: int) -> DataLoader:
    transforms = SAM2Transforms(resolution=resolution, mask_threshold=0.0)
    dataset = build_frame_dataset(
        root=cfg.root,
        split=cfg.split,
        transform=transforms,
        sample_count=cfg.sample_count,
        scan_config=cfg.scan,
    )
    try:
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    except PermissionError as exc:  # pragma: no cover - sandbox fallback
        if cfg.num_workers > 0:
            print(f"[WARN] DataLoader workers={cfg.num_workers} failed ({exc}); retrying with workers=0.")
            return DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
                num_workers=0,
                pin_memory=False,
            )
        raise


def _default_feature_modules(model: nn.Module, limit: int) -> List[str]:
    modules = [name for name, mod in model.named_modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    if not modules:
        return []
    return modules[:limit]


def _run_forward(
    teacher: nn.Module,
    student: nn.Module,
    images: torch.Tensor,
    feature_modules: List[str],
    loss_type: str,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    teacher.eval()
    student.train()

    with torch.no_grad():
        with capture_features(teacher, feature_modules, detach=True) as teacher_feats:
            _ = teacher.forward_image(images)
    with capture_features(student, feature_modules, detach=False) as student_feats:
        _ = student.forward_image(images)

    t_feats = {name: captures for name, captures in teacher_feats.items() if captures}
    s_feats = {name: captures for name, captures in student_feats.items() if captures}
    loss, per_layer = compute_distillation_loss(
        t_feats,
        s_feats,
        loss_type=loss_type,
    )
    return loss, per_layer


def _evaluate(
    teacher: nn.Module,
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_modules: List[str],
    loss_type: str,
    *,
    progress_desc: Optional[str] = None,
) -> Tuple[float, Dict[str, float]]:
    student.eval()
    teacher.eval()
    total_loss = 0.0
    counts = 0
    layer_totals: Dict[str, List[float]] = defaultdict(list)
    iterator = loader
    if progress_desc:
        iterator = tqdm(loader, desc=progress_desc, leave=False)
    with torch.no_grad():
        for batch in iterator:
            images = batch["image"]
            if isinstance(images, torch.Tensor):
                images = images.to(device=device, dtype=torch.float32)
            else:
                continue
            loss, per_layer = _run_forward(teacher, student, images, feature_modules, loss_type)
            total_loss += float(loss.item())
            counts += 1
            for name, value in per_layer.items():
                layer_totals[name].append(float(value.item()))
    avg_layer = {name: float(np.mean(values)) for name, values in layer_totals.items()} if layer_totals else {}
    return total_loss / max(counts, 1), avg_layer


def _apply_bit_reduction(
    teacher: nn.Module,
    student: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    feature_modules: List[str],
    loss_type: str,
    layer_candidates: List[str],
    min_bits: int,
    impact_threshold: float,
) -> Tuple[List[str], List[Dict[str, object]]]:
    logs: List[Dict[str, object]] = []
    if not layer_candidates:
        return [], logs
    quant_layers = collect_quantized_layers(student)
    base_loss, _ = _evaluate(
        teacher,
        student,
        val_loader,
        device,
        feature_modules,
        loss_type,
        progress_desc="Bit reduction baseline",
    )
    accepted: List[str] = []
    for name in layer_candidates:
        module = quant_layers.get(name)
        if module is None:
            continue
        original_bits = module.bit_width
        if original_bits <= min_bits:
            logs.append(
                {
                    "layer": name,
                    "prev_bits": original_bits,
                    "new_bits": original_bits,
                    "loss": float(base_loss),
                    "increase": 0.0,
                    "status": "skipped (min_bits)",
                }
            )
            continue
        module.bit_width = original_bits - 1
        eval_loss, _ = _evaluate(
            teacher,
            student,
            val_loader,
            device,
            feature_modules,
            loss_type,
            progress_desc=f"Eval bit {name}",
        )
        increase = (eval_loss - base_loss) / max(base_loss, 1e-8)
        if eval_loss <= base_loss * (1.0 + impact_threshold):
            accepted.append(name)
            base_loss = eval_loss
            logs.append(
                {
                    "layer": name,
                    "prev_bits": original_bits,
                    "new_bits": module.bit_width,
                    "loss": float(eval_loss),
                    "increase": float(increase),
                    "status": "accepted",
                }
            )
        else:
            module.bit_width = original_bits
            logs.append(
                {
                    "layer": name,
                    "prev_bits": original_bits,
                    "new_bits": original_bits - 1,
                    "loss": float(eval_loss),
                    "increase": float(increase),
                    "status": "rejected",
                }
            )
    return accepted, logs


def train_cdpq(cfg: TrainingConfig) -> Dict[str, object]:
    _init_wandb(cfg)
    device = ensure_device(cfg.student.device)

    base_dir = Path(cfg.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stub = cfg.run_name or "cdpq"
    safe_stub = re.sub(r"[^A-Za-z0-9_.-]", "_", stub).strip("_")
    run_name = f"{timestamp}_{safe_stub}" if safe_stub else timestamp
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"cdpq_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    logger.info("Experiment directory: %s", run_dir)

    teacher = build_sam2_model(
        cfg.teacher.config_name,
        checkpoint_path=cfg.teacher.checkpoint,
        device=cfg.teacher.device,
        hydra_overrides=cfg.teacher.hydra_overrides,
        mode="eval",
    )
    teacher.to(device)
    student = build_sam2_model(
        cfg.student.config_name,
        checkpoint_path=cfg.student.checkpoint,
        device=cfg.student.device,
        hydra_overrides=cfg.student.hydra_overrides,
        mode="train",
    )
    student.to(device)

    wrap_model_for_quantization(
        student,
        default_bits=cfg.quant.initial_bits,
        per_channel=cfg.quant.per_channel,
        skip_prefixes=cfg.quant.skip_prefixes,
    )
    quant_layers = collect_quantized_layers(student)
    feature_modules = (
        cfg.distill.feature_modules
        if cfg.distill.feature_modules
        else _default_feature_modules(student, cfg.distill.feature_limit)
    )

    resolution = getattr(student, "image_size", 512)
    train_loader = _build_dataloader(cfg.train_data, resolution=resolution)
    val_loader = _build_dataloader(cfg.val_data, resolution=resolution)

    optimizer = AdamW(student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    layer_loss_history: Dict[str, List[float]] = defaultdict(list)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        logger.info("[Epoch %s/%s] training...", epoch, cfg.epochs)
        student.train()
        epoch_loss = 0.0
        batch_count = 0
        train_iterator = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}/{cfg.epochs}",
            leave=False,
        )
        avg_bit = sum(module.bit_width for module in quant_layers.values()) / max(len(quant_layers), 1)
        for batch_idx, batch in enumerate(train_iterator, start=1):
            images = batch["image"]
            if not isinstance(images, torch.Tensor):
                continue
            images = images.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, layer_losses = _run_forward(
                teacher,
                student,
                images,
                feature_modules,
                cfg.distill.loss_type,
            )
            loss.backward()
            if cfg.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
            optimizer.step()

            epoch_loss += float(loss.item())
            batch_count += 1
            for name, tensor in layer_losses.items():
                layer_loss_history[name].append(float(tensor.item()))
            avg_bit = (
                sum(module.bit_width for module in quant_layers.values())
                / max(len(quant_layers), 1)
            )
            train_iterator.set_postfix(loss=f"{loss.item():.4f}", avg_bits=f"{avg_bit:.2f}")
            wandb.log({"train/loss": float(loss.item()), "step": global_step, "epoch": epoch})
            wandb.log({"train/avg_bit_width": avg_bit, "step": global_step})
            global_step += 1

            if batch_idx % cfg.log_interval == 0:
                logger.info(
                    "[Epoch %s/%s][Batch %s] loss=%.4f avg_bits=%.2f",
                    epoch,
                    cfg.epochs,
                    batch_idx,
                    float(loss.item()),
                    avg_bit,
                )
        train_iterator.close()
        mean_train_loss = epoch_loss / max(batch_count, 1)
        logger.info(
            "[Epoch %s/%s] train_loss=%.4f, avg_bits=%.2f",
            epoch,
            cfg.epochs,
            mean_train_loss,
            avg_bit,
        )

        if epoch % cfg.eval_interval == 0:
            val_loss, val_layers = _evaluate(
                teacher,
                student,
                val_loader,
                device,
                feature_modules,
                cfg.distill.loss_type,
                progress_desc=f"Eval epoch {epoch}/{cfg.epochs}",
            )
            logger.info(
                "[Epoch %s/%s] val_loss=%.4f",
                epoch,
                cfg.epochs,
                val_loss,
            )
            wandb.log({"val/loss": val_loss, "epoch": epoch})
            bit_lookup = {name: module.bit_width for name, module in quant_layers.items()}
            if cfg.distill.confidence_method == "variance":
                candidates = confidence_variance(
                    layer_loss_history,
                    bit_lookup,
                    mean_threshold=cfg.distill.layer_threshold,
                    var_threshold=cfg.distill.variance_threshold,
                    min_bits=cfg.quant.min_bits,
                )
            else:
                candidates = confidence_baseline(
                    layer_loss_history,
                    bit_lookup,
                    threshold=cfg.distill.layer_threshold,
                    min_bits=cfg.quant.min_bits,
                )

            if cfg.distill.target_roots:
                filtered = []
                for name in candidates:
                    if any(name.startswith(root) for root in cfg.distill.target_roots):
                        filtered.append(name)
                logger.info(
                    "[Epoch %s/%s] filtering candidates by roots %s: %s -> %s",
                    epoch,
                    cfg.epochs,
                    cfg.distill.target_roots,
                    len(candidates),
                    len(filtered),
                )
                candidates = filtered

            logger.info(
                "[Epoch %s/%s] bit reduction candidates=%s",
                epoch,
                cfg.epochs,
                len(candidates),
            )

            accepted, attempt_logs = _apply_bit_reduction(
                teacher,
                student,
                val_loader,
                device,
                feature_modules,
                cfg.distill.loss_type,
                candidates,
                cfg.quant.min_bits,
                cfg.distill.impact_threshold,
            )
            if attempt_logs:
                for entry in attempt_logs:
                    logger.info(
                        "Bit reduction attempt | layer=%s | prev_bits=%s | new_bits=%s | eval_loss=%.6f | increase=%.4f | status=%s",
                        entry["layer"],
                        entry["prev_bits"],
                        entry["new_bits"],
                        entry["loss"],
                        entry["increase"],
                        entry["status"],
                    )
            avg_increase = (
                np.mean([log["increase"] for log in attempt_logs if log["status"] == "accepted"])
                if accepted
                else 0.0
            )
            wandb.log(
                {
                    "quant/updated_layers": len(accepted),
                    "quant/accept_rate": len(accepted) / max(len(candidates), 1),
                    "quant/candidates": len(candidates),
                    "quant/mean_increase": float(avg_increase),
                    "epoch": epoch,
                }
            )
            if accepted:
                logger.info(
                    "[Epoch %s/%s] bit reduction accepted layers: %s",
                    epoch,
                    cfg.epochs,
                    accepted,
                )
            else:
                logger.info("[Epoch %s/%s] bit reduction skipped", epoch, cfg.epochs)
            avg_bit = (
                sum(module.bit_width for module in quant_layers.values())
                / max(len(quant_layers), 1)
            )
            logger.info(
                "[Epoch %s/%s] avg_bits(after reduction)=%.2f",
                epoch,
                cfg.epochs,
                avg_bit,
            )

    float_ckpt_path = run_dir / "student_quantized_float.pt"
    torch.save({"model": student.state_dict()}, float_ckpt_path)
    logger.info("Saved float checkpoint to %s", float_ckpt_path)

    from ..io.storage import save_package, serialize_targets

    quant_package = serialize_targets(
        quant_layers,
        model_config=cfg.student.config_name,
        checkpoint_path=cfg.student.checkpoint,
        metadata={
            "final_bitwidths": {name: module.bit_width for name, module in quant_layers.items()},
        },
    )
    quant_pkg_path = run_dir / "student_quantized_package.pt"
    save_package(quant_package, quant_pkg_path)
    logger.info("Saved quantized package to %s", quant_pkg_path)

    results = {
        "run_dir": str(run_dir),
        "float_checkpoint": str(float_ckpt_path),
        "quantized_package": str(quant_pkg_path),
        "final_bitwidths": {name: module.bit_width for name, module in quant_layers.items()},
    }
    logger.info("Training complete. Artifacts stored in %s", run_dir)
    return results
