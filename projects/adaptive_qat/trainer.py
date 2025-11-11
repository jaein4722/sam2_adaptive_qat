"""Project-local trainer with experiment-specific instrumentation."""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional

import torch

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig, OmegaConf

from training.trainer import (  # type: ignore[import]
    Phase,
    Trainer,
    print_model_summary,
    unwrap_ddp_if_wrapped,
)
from training.utils.checkpoint_utils import load_state_dict_into_model
from training.utils.distributed import barrier
from training.utils.train_utils import (  # type: ignore[import]
    AverageMeter,
    MemMeter,
    ProgressMeter,
)

from .utils.logging import instantiate_logger


def _unique_sorted_fractions(fractions: Optional[Iterable[float]]) -> List[float]:
    if not fractions:
        return []
    filtered: List[float] = []
    for value in fractions:
        try:
            val = float(value)
        except (TypeError, ValueError):
            logging.warning("Ignoring invalid mid-epoch validation fraction: %s", value)
            continue
        if not 0.0 < val < 1.0:
            logging.warning(
                "Mid-epoch validation fraction %.3f is outside (0, 1); ignoring.", val
            )
            continue
        filtered.append(val)
    if not filtered:
        return []
    # Preserve order but drop duplicates, then sort for monotonic scheduling
    ordered = list(dict.fromkeys(filtered))
    ordered.sort()
    return ordered


class AdaptiveTrainer(Trainer):
    """Extends the stock trainer with mid-epoch validation and W&B logging."""

    def __init__(
        self,
        *,
        logging: Dict[str, Any],
        mid_epoch_val_fractions: Optional[Iterable[float]] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(logging, DictConfig):
            logging_conf = OmegaConf.to_container(logging, resolve=True)
        else:
            logging_conf = dict(logging)
        self._wandb_conf = logging_conf.pop("wandb", None)
        self._mid_epoch_val_fractions_cfg = _unique_sorted_fractions(
            mid_epoch_val_fractions
        )

        super().__init__(logging=logging_conf, **kwargs)

    # ------------------------------------------------------------------
    def _setup_components(self):
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = self._collect_val_keys(self.data_conf[val_phase])
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}

        self.mid_epoch_val_fractions = self._mid_epoch_val_fractions_cfg
        self.logger = instantiate_logger(
            self.logging_conf, wandb_conf=self._wandb_conf, rank=self.distributed_rank
        )
        self.model = instantiate(self.model_conf, _convert_="all")
        print_model_summary(self.model)

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = torch.nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )

        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _update_quantization_progress(self, progress: float) -> None:
        model_module = unwrap_ddp_if_wrapped(self.model)
        bit_controller = getattr(model_module, "bit_controller", None)
        if bit_controller is not None:
            bit_controller.update_progress(progress)
        if self.loss is None:
            return
        for module in self.loss.values():
            if hasattr(module, "set_progress"):
                module.set_progress(progress)

    # ------------------------------------------------------------------
    def train_epoch(self, train_loader):
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)
        next_mid_val_idx = 0

        loss_names = [f"Losses/{phase}_{key}_loss" for key in self.loss.keys()]
        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts: Dict[str, AverageMeter] = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            batch = batch.to(self.device, non_blocking=True)
            grads_unscaled = False

            exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
            progress_ratio = float(exact_epoch) / self.max_epochs
            self._update_quantization_progress(progress_ratio)

            try:
                super()._run_step(batch, phase, loss_mts, extra_loss_mts)

                self.scaler.unscale_(self.optim.optimizer)
                grads_unscaled = True

                self.where = progress_ratio
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        "Skipping scheduler update since training is at the end (where=%.3f).",
                        self.where,
                    )

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = f"{j}_" if len(self.optim.optimizer.param_groups) > 1 else ""
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                if self.gradient_clipper is not None:
                    if not grads_unscaled:
                        self.scaler.unscale_(self.optim.optimizer)
                        grads_unscaled = True
                    self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

                if (
                    self.val_dataset is not None
                    and next_mid_val_idx < len(self.mid_epoch_val_fractions)
                ):
                    progress_ratio = float(data_iter + 1) / max(iters_per_epoch, 1)
                    while (
                        next_mid_val_idx < len(self.mid_epoch_val_fractions)
                        and progress_ratio
                        >= self.mid_epoch_val_fractions[next_mid_val_idx] - self.EPSILON
                    ):
                        fraction = self.mid_epoch_val_fractions[next_mid_val_idx]
                        self._run_mid_epoch_validation(
                            fraction=fraction,
                            iteration=data_iter + 1,
                            total_iters=iters_per_epoch,
                        )
                        next_mid_val_idx += 1

            except RuntimeError as exc:
                message = str(exc).lower()
                if "nan" in message or "inf" in message:
                    logging.error(
                        "AdaptiveTrainer detected invalid gradients (%s); skipping step",
                        exc,
                    )
                    self.optim.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    end = time.time()
                    continue
                raise
            except FloatingPointError:
                raise

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    # ------------------------------------------------------------------
    def _run_mid_epoch_validation(
        self, *, fraction: float, iteration: int, total_iters: int
    ) -> None:
        if not self.val_dataset:
            return

        logging.info(
            "Running mid-epoch validation at epoch %.3f (fraction %.3f, iteration %d/%d)",
            float(self.epoch),
            fraction,
            iteration,
            total_iters,
        )
        barrier()

        previous_best = dict(self.best_meter_values)

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        try:
            outs = self.val_epoch(dataloader, phase=Phase.VAL)
        finally:
            del dataloader
            gc.collect()

        outs["Trainer/epoch_fraction"] = fraction
        outs["Trainer/iteration_within_epoch"] = iteration
        outs["Trainer/iterations_per_epoch"] = total_iters

        self.logger.log_dict(outs, self.steps[Phase.VAL])

        if self.distributed_rank != 0:
            return

        serialisable = {
            "epoch": float(self.epoch),
            "fraction": float(fraction),
            "iteration": int(iteration),
            "total_iterations": int(total_iters),
        }
        for key, value in outs.items():
            if key in serialisable:
                continue
            scalar = _float_value(value)
            if scalar is not None:
                serialisable[key] = scalar

        stats_path = os.path.join(self.logging_conf.log_dir, "mid_val_stats.json")
        with g_pathmgr.open(stats_path, "a") as f:
            f.write(json.dumps(serialisable) + "\n")

        self._maybe_update_best_checkpoint(
            self.epoch + fraction, outs, previous_best=previous_best
        )

    # ------------------------------------------------------------------
    def _maybe_update_best_checkpoint(
        self,
        epoch_marker: float,
        val_outputs: Dict[str, Any],
        previous_best: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reserializes best checkpoints and stats when mid-epoch val improves meters."""

        previous_best = previous_best or {}
        improved_keys = [
            key
            for key, value in self.best_meter_values.items()
            if previous_best.get(key) != value
        ]
        if not improved_keys:
            return

        trainer_where = _float_value(val_outputs.get("Trainer/where"))
        if trainer_where is None:
            trainer_where = float(self.where)

        trainer_state = {
            "Trainer/epoch": float(epoch_marker),
            "Trainer/where": trainer_where,
            "Trainer/steps_train": self.steps.get(Phase.TRAIN),
            "Trainer/steps_val": self.steps.get(Phase.VAL),
        }

        # Drop items that cannot be serialized downstream.
        trainer_state = {
            key: value
            for key, value in trainer_state.items()
            if value is not None
        }
        self.best_meter_values.update(trainer_state)

        checkpoint_names: List[str] = []
        if self.checkpoint_conf.save_best_meters:
            for tracked_key in improved_keys:
                meter_group = tracked_key.rsplit("/", 1)[0]
                if meter_group in self.checkpoint_conf.save_best_meters:
                    checkpoint_names.append(tracked_key.replace("/", "_"))

        if checkpoint_names:
            logging.info(
                "Mid-epoch validation improved %s; saving checkpoints at epoch %.3f",
                ", ".join(checkpoint_names),
                epoch_marker,
            )
            self.save_checkpoint(epoch_marker, checkpoint_names)

        serialisable = dict(trainer_state)
        for tracked_key in improved_keys:
            scalar = _float_value(self.best_meter_values.get(tracked_key))
            if scalar is not None:
                serialisable[tracked_key] = scalar

        if not serialisable:
            return

        stats_path = os.path.join(self.logging_conf.log_dir, "best_stats.json")
        with g_pathmgr.open(stats_path, "a") as f:
            f.write(json.dumps(serialisable) + "\n")

    # ------------------------------------------------------------------
    def _collect_val_keys(self, datasets_conf: Any) -> List[str]:
        from training.utils.train_utils import collect_dict_keys

        return collect_dict_keys(datasets_conf)

    # ------------------------------------------------------------------
    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """Persist checkpoints with an inference-ready student state dict."""

        if checkpoint is None:
            super()._save_checkpoint(checkpoint, checkpoint_path)
            return

        checkpoint_to_save = dict(checkpoint)
        full_state = checkpoint_to_save.get("model", {})

        base_model = unwrap_ddp_if_wrapped(self.model)
        student_prefix = checkpoint_to_save.get("model_student_prefix", "student.")
        has_prefixed_student = any(
            isinstance(key, str) and key.startswith(student_prefix) for key in full_state
        )

        if has_prefixed_student:
            student_state = {
                key[len(student_prefix) :]: value
                for key, value in full_state.items()
                if key.startswith(student_prefix)
            }
            aux_state = {
                key: value
                for key, value in full_state.items()
                if not key.startswith(student_prefix)
            }
        else:
            student_model = unwrap_ddp_if_wrapped(getattr(base_model, "student", base_model))
            student_state = student_model.state_dict()
            aux_state = full_state

        sanitize_fn = getattr(base_model, "sanitize_student_state_dict", None)
        if callable(sanitize_fn):
            student_state = sanitize_fn(student_state)

        checkpoint_to_save["model"] = student_state
        checkpoint_to_save["model_aux"] = aux_state
        checkpoint_to_save["model_student_prefix"] = "student."
        checkpoint_to_save.setdefault("checkpoint_format", "adaptive_qat_v2")

        super()._save_checkpoint(checkpoint_to_save, checkpoint_path)

    # ------------------------------------------------------------------
    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        student_prefix = checkpoint.get("model_student_prefix", "student.")
        student_state = checkpoint.get("model")
        aux_state = checkpoint.get("model_aux")

        if student_state is None:
            raise KeyError(
                "Checkpoint does not contain a 'model' entry; cannot resume training."
            )

        if isinstance(student_state, dict):
            full_state = {
                f"{student_prefix}{key}": value for key, value in student_state.items()
            }
        else:
            full_state = student_state

        if isinstance(aux_state, dict):
            full_state.update(aux_state)

        load_state_dict_into_model(
            model=self.model,
            state_dict=full_state,
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            try:
                self.optim.optimizer.load_state_dict(optimizer_state)
            except ValueError as exc:
                logging.warning(
                    "AdaptiveTrainer detected optimizer param-group mismatch when resuming "
                    "from %s; optimizer state will be reinitialised. Details: %s",
                    ckpt_path,
                    exc,
                )
        else:
            logging.warning(
                "AdaptiveTrainer resume checkpoint %s does not contain an optimizer "
                "state; continuing with freshly initialised optimizer.",
                ckpt_path,
            )
        if self.loss is not None and checkpoint.get("loss") is not None:
            self.loss.load_state_dict(checkpoint["loss"], strict=True)

        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_meter_values = checkpoint.get("best_meter_values", {})

        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])


def _float_value(value: Any) -> Optional[float]:
    import numpy as np

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().mean().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.mean())
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
