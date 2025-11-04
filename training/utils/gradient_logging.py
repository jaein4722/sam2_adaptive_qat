"""Utilities to export gradients during training."""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import torch
from iopath.common.file_io import g_pathmgr

from training.utils.train_utils import makedir


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    ddp_cls = getattr(torch.nn.parallel, "DistributedDataParallel", None)
    if ddp_cls is not None and isinstance(model, ddp_cls):
        return model.module
    return model


class LayerGradientLogger:
    """Collects per-parameter gradient statistics at a fixed interval.

    The logger is instantiated from a Hydra config and called from the trainer
    after gradients are computed and before the optimizer step. When the
    configured interval is reached (based on the global training step), the
    logger dumps a JSON summary with statistics for every parameter that has a
    gradient. Optionally, the full gradient tensors can also be persisted to a
    PyTorch checkpoint file for detailed offline inspection. When weight
    snapshots are enabled, the logger additionally writes the model weights for
    the same step, ensuring gradients and parameters stay in sync.
    """

    _SUPPORTED_STATS = {
        "l2_norm",
        "mean",
        "std",
        "max",
        "min",
        "abs_max",
    }

    def __init__(
        self,
        output_dir: str,
        log_every: int = 100,
        stats: Optional[List[str]] = None,
        save_full_tensor: bool = False,
        save_weights: bool = True,
        weights_output_dir: Optional[str] = None,
    ) -> None:
        if log_every <= 0:
            raise ValueError("log_every must be a positive integer")
        self.output_dir = output_dir
        self.log_every = int(log_every)
        self.save_full_tensor = save_full_tensor
        self.save_weights = bool(save_weights)
        self.weights_output_dir = (
            weights_output_dir or os.path.join(self.output_dir, "weights")
        )
        self._last_logged_step: Optional[int] = None

        if stats is None:
            stats = ["l2_norm", "mean", "std", "max", "min"]
        invalid = set(stats) - self._SUPPORTED_STATS
        if invalid:
            raise ValueError(f"Unsupported stats requested: {sorted(invalid)}")
        self.stats = stats

        makedir(self.output_dir)
        if self.save_weights:
            makedir(self.weights_output_dir)
        logging.info(
            "Gradient logger initialised — exporting every %s steps to %s",
            self.log_every,
            self.output_dir,
        )
        if self.save_weights:
            logging.info(
                "Weight snapshots will be stored at %s",
                self.weights_output_dir,
            )

    def rebase_to_checkpoint_dir(
        self,
        checkpoint_dir: Optional[str],
        *,
        weight_subdir: str = "step_weights",
    ) -> None:
        """Place weight snapshots under the checkpoint root without moving logs."""
        if not (checkpoint_dir and self.save_weights):
            return

        new_weights_dir = os.path.join(checkpoint_dir, weight_subdir)
        if new_weights_dir == self.weights_output_dir:
            return

        self.weights_output_dir = new_weights_dir
        makedir(self.weights_output_dir)

        logging.info(
            "Gradient weight snapshots will be stored at %s",
            self.weights_output_dir,
        )

    def __call__(
        self,
        model: torch.nn.Module,
        step: int,
        rank: int,
        where: float,
    ) -> None:
        """Export gradient information if the step hits the configured interval."""

        if rank != 0:
            # Avoid redundant exports from non-zero ranks in distributed runs.
            return

        if step <= 0 or step % self.log_every != 0:
            return

        if self._last_logged_step == step:
            return

        gradients_summary: Dict[str, Dict[str, object]] = {}
        gradients_full: Dict[str, torch.Tensor] = {}
        total_parameters = 0
        parameters_with_grad = 0

        for name, param in model.named_parameters():
            total_parameters += 1
            grad = param.grad
            if grad is None:
                gradients_summary[name] = {
                    "has_grad": False,
                    "dtype": str(param.dtype).replace("torch.", ""),
                }
                continue

            parameters_with_grad += 1
            grad_cpu = grad.detach().to(device="cpu", dtype=torch.float32)
            stats_payload: Dict[str, float] = {}

            if "l2_norm" in self.stats:
                stats_payload["l2_norm"] = float(torch.linalg.vector_norm(grad_cpu).item())
            if "mean" in self.stats:
                stats_payload["mean"] = float(grad_cpu.mean().item())
            if "std" in self.stats:
                stats_payload["std"] = float(grad_cpu.std(unbiased=False).item())
            if "max" in self.stats:
                stats_payload["max"] = float(grad_cpu.max().item())
            if "min" in self.stats:
                stats_payload["min"] = float(grad_cpu.min().item())
            if "abs_max" in self.stats:
                stats_payload["abs_max"] = float(grad_cpu.abs().max().item())

            gradients_summary[name] = {
                "has_grad": True,
                "dtype": str(grad.dtype).replace("torch.", ""),
                "shape": list(grad.shape),
                "numel": grad.numel(),
                "stats": stats_payload,
            }

            if self.save_full_tensor:
                gradients_full[name] = grad.detach().cpu()

        timestamp = datetime.utcnow().isoformat() + "Z"
        json_record = {
            "step": step,
            "where": where,
            "timestamp": timestamp,
            "log_every": self.log_every,
            "total_parameters": total_parameters,
            "parameters_with_grad": parameters_with_grad,
            "parameters_without_grad": total_parameters - parameters_with_grad,
            "gradients": gradients_summary,
        }

        tensor_relative_path = None
        if self.save_full_tensor and gradients_full:
            tensor_filename = f"gradients_step_{step:08d}.pt"
            tensor_path = os.path.join(self.output_dir, tensor_filename)
            with g_pathmgr.open(tensor_path, "wb") as handle:
                torch.save(gradients_full, handle)
            tensor_relative_path = tensor_filename
            json_record["tensor_file"] = tensor_relative_path

        weights_relative_path = self._save_weights_snapshot(model, step, where)
        if weights_relative_path is not None:
            json_record["weights_file"] = weights_relative_path

        json_filename = f"gradients_step_{step:08d}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        with g_pathmgr.open(json_path, "w") as handle:
            json.dump(json_record, handle, indent=2)

        logging.info(
            "Stored gradient statistics for step %s (%s gradients with data) to %s",
            step,
            parameters_with_grad,
            json_path,
        )
        if tensor_relative_path:
            logging.info("Full gradient tensors saved to %s", tensor_relative_path)

        self._last_logged_step = step

    def _save_weights_snapshot(
        self,
        model: torch.nn.Module,
        step: int,
        where: float,
    ) -> Optional[str]:
        if not self.save_weights:
            return None

        filename = f"weights_step_{step:08d}.pt"
        weights_path = os.path.join(self.weights_output_dir, filename)
        unwrapped = _unwrap_model(model)
        with torch.no_grad():
            payload = {
                "model": {
                    name: tensor.detach().cpu()
                    for name, tensor in unwrapped.state_dict().items()
                },
                "step": step,
                "where": where,
            }
        with g_pathmgr.open(weights_path, "wb") as handle:
            torch.save(payload, handle)
        del payload
        logging.info("Stored weight snapshot for step %s to %s", step, weights_path)
        try:
            rel_path = os.path.relpath(weights_path, self.output_dir)
        except ValueError:
            rel_path = weights_path
        return rel_path
