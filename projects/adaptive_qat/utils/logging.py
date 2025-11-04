"""Custom logging utilities for adaptive QAT experiments.

This module keeps all experiment-specific instrumentation inside the project
directory so that upstream SAM2 code remains untouched.
"""

from __future__ import annotations

import atexit
import logging
from typing import Any, Dict, Optional

from hydra.utils import instantiate

from omegaconf import DictConfig, ListConfig

from training.utils.logger import Logger as TorchLogger

_ACTIVE_WANDB_RUN = None


def _is_hydra_config(value: Any) -> bool:
    return isinstance(value, (DictConfig, ListConfig))


def _to_plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, ListConfig):
        return [_to_plain(v) for v in value]
    return value


class AdaptiveLogger:
    """Wraps the default SAM2 logger and adds optional Weights & Biases logging."""

    def __init__(
        self,
        logging_conf,
        *,
        wandb_conf: Optional[Dict[str, Any]] = None,
        rank: int = 0,
    ) -> None:
        self._base_logger = TorchLogger(logging_conf)
        self._rank = rank
        self._wandb_run = self._init_wandb(wandb_conf) if wandb_conf else None

    def _init_wandb(self, wandb_conf: Dict[str, Any]):
        global _ACTIVE_WANDB_RUN

        if self._rank != 0:
            logging.debug("Skipping wandb init on non-zero rank %s", self._rank)
            return None

        if _ACTIVE_WANDB_RUN is not None:
            logging.debug("Reusing existing wandb run: %s", _ACTIVE_WANDB_RUN.name)
            return _ACTIVE_WANDB_RUN

        if wandb_conf is None:
            return None

        try:
            import wandb  # type: ignore[import]
        except ModuleNotFoundError:
            logging.warning("wandb is not installed; wandb logging disabled.")
            return None

        init_kwargs = _to_plain(wandb_conf)
        if not isinstance(init_kwargs, dict):
            logging.warning("Unexpected wandb config type %s; logging disabled.", type(init_kwargs))
            return None

        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        try:
            run = wandb.init(**init_kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("Failed to initialise wandb (%s); logging disabled.", exc)
            return None

        _ACTIVE_WANDB_RUN = run
        atexit.register(wandb.finish)
        logging.info("wandb run initialised: %s", run.name)
        return run

    # Proxy methods -----------------------------------------------------
    def log_dict(self, payload: Dict[str, Any], step: Optional[int]) -> None:
        self._base_logger.log_dict(payload, step)
        if self._wandb_run is None:
            return
        log_payload = {}
        for key, value in payload.items():
            scalar = _scalar_value(value)
            if scalar is not None:
                log_payload[key] = scalar
        if log_payload:
            self._wandb_run.log(log_payload, step=step)

    def log(self, name: str, data: Any, step: Optional[int]) -> None:
        self._base_logger.log(name, data, step)
        if self._wandb_run is None:
            return
        scalar = _scalar_value(data)
        if scalar is None:
            return
        self._wandb_run.log({name: scalar}, step=step)

    def log_hparams(self, hparams: Dict[str, Any], meters: Dict[str, Any]) -> None:
        self._base_logger.log_hparams(hparams, meters)
        if self._wandb_run is None:
            return

        flat_hparams = {
            key: value
            for key, value in ((k, _scalar_value(v)) for k, v in hparams.items())
            if value is not None
        }
        if flat_hparams:
            self._wandb_run.config.update(flat_hparams, allow_val_change=True)

        flat_meters = {
            key: value
            for key, value in ((k, _scalar_value(v)) for k, v in meters.items())
            if value is not None
        }
        if flat_meters:
            self._wandb_run.log({f"hparam/{k}": v for k, v in flat_meters.items()})

    # Convenience proxies ------------------------------------------------
    @property
    def tb_logger(self):  # pragma: no cover - passthrough for existing call-sites
        return getattr(self._base_logger, "tb_logger", None)


def _scalar_value(value: Any) -> Optional[float]:
    import numpy as np
    import torch

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


def instantiate_logger(logging_conf, *, wandb_conf: Optional[Dict[str, Any]], rank: int):
    """Hydra-friendly construction helper."""

    # Hydra instantiate on a DictConfig needs explicit conversion
    if _is_hydra_config(logging_conf):
        logging_conf = instantiate(logging_conf)
    return AdaptiveLogger(logging_conf, wandb_conf=wandb_conf, rank=rank)

