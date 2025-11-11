"""Thin wrapper around SAM2Train that injects min-max fake-quant hooks."""

from __future__ import annotations

import logging
from typing import Mapping, Optional

import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from training.utils.checkpoint_utils import (
    load_checkpoint_and_apply_kernels,
    load_state_dict_into_model,
)

from ..quantization import MinMaxQuantConfig, apply_minmax_quantization


class MinMaxQATModel(nn.Module):
    """Wraps a SAM2 student model and inserts min-max fake-quant modules."""

    def __init__(
        self,
        *,
        model: Mapping | nn.Module | None = None,
        student: Mapping | nn.Module | None = None,
        quantization: MinMaxQuantConfig | DictConfig,
        checkpoint_path: Optional[str] = None,
        student_checkpoint: Optional[str] = None,
        strict_checkpoint_load: bool = True,
    ) -> None:
        super().__init__()

        base_model_cfg = student if student is not None else model
        if base_model_cfg is None:
            raise ValueError("Either `model` or `student` must be provided.")

        if isinstance(base_model_cfg, nn.Module):
            base_model = base_model_cfg
        else:
            base_model = instantiate(base_model_cfg)

        ckpt_path = checkpoint_path if checkpoint_path is not None else student_checkpoint
        if ckpt_path:
            logging.info("[minmax_qat] Loading baseline checkpoint from %s", ckpt_path)
            state = load_checkpoint_and_apply_kernels(
                checkpoint_path=ckpt_path,
                ckpt_state_dict_keys=["model"],
            )
            load_state_dict_into_model(
                state,
                base_model,
                strict=strict_checkpoint_load,
                ignore_missing_keys=None,
                ignore_unexpected_keys=None,
            )

        if isinstance(quantization, DictConfig):
            quant_dict = OmegaConf.to_container(quantization, resolve=True)
            target = quant_dict.pop("_target_", None)
            if target and target != f"{MinMaxQuantConfig.__module__}.{MinMaxQuantConfig.__name__}":
                logging.warning(
                    "[minmax_qat] quantization target %s does not match MinMaxQuantConfig; proceeding with field mapping",
                    target,
                )
            quant_cfg = MinMaxQuantConfig(**quant_dict)
        else:
            quant_cfg = quantization
        if not isinstance(quant_cfg, MinMaxQuantConfig):
            raise TypeError("quantization config must be a MinMaxQuantConfig instance")

        replaced = apply_minmax_quantization(base_model, quant_cfg)
        logging.info(
            "[minmax_qat] Inserted fake-quant modules for %d layers", len(replaced)
        )

        self.model = base_model

    def forward(self, batch):  # type: ignore[override]
        return self.model(batch)
