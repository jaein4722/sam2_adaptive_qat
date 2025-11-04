"""Entry point for adaptive QAT training built on the SAM 2 trainer."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from training.train import add_pythonpath_to_sys_path, single_node_runner  # noqa: E402
from training.utils.train_utils import makedir, register_omegaconf_resolvers  # noqa: E402


DEFAULT_CONFIG = "projects/adaptive_qat/configs/adaptive_qat_template.yaml"

def _resolve_experiment_dir(cfg, override: str | None) -> str:
    if override:
        cfg.paths.experiment_dir = override
    exp_dir = os.path.abspath(cfg.paths.experiment_dir)
    cfg.launcher.experiment_log_dir = exp_dir
    cfg.trainer.logging.log_dir = os.path.join(exp_dir, "logs")
    cfg.trainer.checkpoint.save_dir = os.path.join(exp_dir, "checkpoints")
    return exp_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive QAT distillation training")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to Hydra config")
    parser.add_argument("--experiment-dir", type=str, help="Optional override for experiment directory")
    parser.add_argument("--main-port", type=int, default=None, help="Port used for distributed init")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    register_omegaconf_resolvers()

    exp_dir = _resolve_experiment_dir(cfg, args.experiment_dir)
    makedir(exp_dir)

    add_pythonpath_to_sys_path()

    print("###################### Adaptive QAT Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("###############################################################")

    with open(os.path.join(exp_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    main_port = args.main_port if args.main_port is not None else random.randint(29500, 29999)
    single_node_runner(cfg, main_port=main_port)


if __name__ == "__main__":
    main()
