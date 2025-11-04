"""Entry-point for fine-tuning SAM 2 on the SA-V dataset."""

import argparse
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Optional
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import submitit
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig, OmegaConf

from training.train import (
    SubmititRunner,
    add_pythonpath_to_sys_path,
    single_node_runner,
)
from training.utils.train_utils import makedir, register_omegaconf_resolvers

from projects.sav_finetune.utils.dataset_preparation import (
    prepare_training_frames_if_needed,
)


DEFAULT_CONFIG = "projects/sav_finetune/configs/sam2.1_hiera_b+_sav_finetune.yaml"
DEFAULT_LOG_ROOT = "sam2_logs"


def _update_if_not_none(cfg: DictConfig, dotted_key: str, value: Optional[str]):
    """Set the value of a nested config key if a non-None value is provided."""
    if value is None:
        return
    current = cfg
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in current or current[key] is None:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _validate_dataset_paths(cfg: DictConfig) -> None:
    missing = []
    if not cfg.dataset.img_folder:
        missing.append("dataset.img_folder")
    if not cfg.dataset.gt_folder:
        missing.append("dataset.gt_folder")
    if missing:
        raise ValueError(
            "Missing required dataset paths: " + ", ".join(missing)
        )


def _validate_validation_paths(cfg: DictConfig) -> bool:
    val_img = OmegaConf.select(cfg, "dataset.val_img_folder")
    val_gt = OmegaConf.select(cfg, "dataset.val_gt_folder")
    if val_img or val_gt:
        if not val_img or not val_gt:
            raise ValueError(
                "Validation requires both dataset.val_img_folder and dataset.val_gt_folder"
            )
        for path in (val_img, val_gt):
            if not g_pathmgr.exists(path):
                raise ValueError(f"Validation path does not exist: {path}")
        val_file_list = OmegaConf.select(cfg, "dataset.val_file_list_txt")
        if val_file_list and not g_pathmgr.exists(val_file_list):
            raise ValueError(
                f"Validation file list does not exist: {val_file_list}"
            )
        val_exclude = OmegaConf.select(cfg, "dataset.val_excluded_videos_list_txt")
        if val_exclude and not g_pathmgr.exists(val_exclude):
            raise ValueError(
                f"Validation exclude list does not exist: {val_exclude}"
            )
        return True
    return False


def _configure_validation(
    cfg: DictConfig,
    *,
    num_val_workers: Optional[int] = None,
) -> bool:
    val_img = OmegaConf.select(cfg, "dataset.val_img_folder")
    val_gt = OmegaConf.select(cfg, "dataset.val_gt_folder")
    if not (val_img and val_gt):
        return False

    logging.info(
        "Enabling validation dataloader with frames from %s and annotations from %s",
        val_img,
        val_gt,
    )
    train_data_cfg = OmegaConf.to_container(
        cfg.trainer.data.train, resolve=False
    )
    val_cfg = deepcopy(train_data_cfg)
    val_cfg["shuffle"] = False
    val_cfg["drop_last"] = False
    if num_val_workers is not None:
        val_cfg["num_workers"] = num_val_workers
    else:
        val_cfg["num_workers"] = 0

    val_transforms = OmegaConf.select(cfg, "vos.val_transforms")
    if val_transforms is None:
        val_transforms = OmegaConf.create(
            [
                {
                    "_target_": "training.dataset.transforms.ComposeAPI",
                    "transforms": [
                        {
                            "_target_": "training.dataset.transforms.RandomResizeAPI",
                            "sizes": cfg.scratch.resolution,
                            "square": True,
                            "consistent_transform": True,
                        },
                        {"_target_": "training.dataset.transforms.ToTensorAPI"},
                        {
                            "_target_": "training.dataset.transforms.NormalizeAPI",
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225],
                        },
                    ],
                }
            ]
        )
    val_transforms_container = OmegaConf.to_container(
        val_transforms, resolve=False
    )

    def _update_dataset_config(node):
        if isinstance(node, dict):
            target = node.get("_target_")
            if target == "training.dataset.vos_dataset.VOSDataset":
                node["training"] = False
                node["transforms"] = deepcopy(val_transforms_container)
                node["multiplier"] = 1
                node["sampler"] = {
                    "_target_": "training.dataset.vos_sampler.EvalSampler"
                }

                val_max_frames = OmegaConf.select(cfg, "dataset.val_max_frames")
                if val_max_frames is None:
                    val_max_frames = min(int(cfg.scratch.num_frames), 4)
                if val_max_frames is not None and val_max_frames > 0:
                    node["sampler"]["max_frames"] = int(val_max_frames)

                val_frame_stride = OmegaConf.select(cfg, "dataset.val_frame_stride")
                if val_frame_stride is None:
                    val_frame_stride = 1
                node["sampler"]["frame_stride"] = max(1, int(val_frame_stride))

                file_list = OmegaConf.select(cfg, "dataset.val_file_list_txt")
                excluded_list = OmegaConf.select(
                    cfg, "dataset.val_excluded_videos_list_txt"
                )

                png_dataset = {
                    "_target_": "training.dataset.vos_raw_dataset.PNGRawDataset",
                    "img_folder": val_img,
                    "gt_folder": val_gt,
                }
                if file_list is not None:
                    png_dataset["file_list_txt"] = file_list
                if excluded_list is not None:
                    png_dataset["excluded_videos_list_txt"] = excluded_list

                override_map = {
                    "sample_rate": "dataset.val_sample_rate",
                    "is_palette": "dataset.val_is_palette",
                    "single_object_mode": "dataset.val_single_object_mode",
                    "truncate_video": "dataset.val_truncate_video",
                    "frames_sampling_mult": "dataset.val_frames_sampling_mult",
                }
                for key, dotted_key in override_map.items():
                    value = OmegaConf.select(cfg, dotted_key)
                    if value is not None:
                        png_dataset[key] = value

                if "is_palette" not in png_dataset:
                    try:
                        entries = g_pathmgr.ls(val_gt)
                    except Exception:
                        try:
                            entries = os.listdir(val_gt)
                        except OSError:
                            entries = []
                    has_nested_mask_dirs = False
                    for entry in entries:
                        if entry.startswith("."):
                            continue
                        entry_path = os.path.join(val_gt, entry)
                        try:
                            if g_pathmgr.isdir(entry_path):
                                has_nested_mask_dirs = True
                                break
                        except Exception:
                            if os.path.isdir(entry_path):
                                has_nested_mask_dirs = True
                                break
                    if has_nested_mask_dirs:
                        logging.info(
                            "Detected per-object mask directories under %s; using MultiplePNGSegmentLoader",
                            val_gt,
                        )
                        png_dataset["is_palette"] = False

                node["video_dataset"] = png_dataset
            else:
                for value in node.values():
                    _update_dataset_config(value)
        elif isinstance(node, list):
            for item in node:
                _update_dataset_config(item)

    _update_dataset_config(val_cfg)

    collate_cfg = val_cfg.get("collate_fn")
    if isinstance(collate_cfg, dict):
        collate_cfg["dict_key"] = "val"

    cfg.trainer.data["val"] = OmegaConf.create(val_cfg)
    cfg.trainer.mode = "train"
    if OmegaConf.select(cfg, "trainer.val_epoch_freq") is None:
        cfg.trainer.val_epoch_freq = 1

    if OmegaConf.select(cfg, "trainer.loss.val") is None:
        loss_all = OmegaConf.select(cfg, "trainer.loss.all")
        if loss_all is not None:
            cfg.trainer.loss["val"] = OmegaConf.create(
                OmegaConf.to_container(loss_all, resolve=False)
            )
    return True


def _print_and_dump_config(cfg: DictConfig, exp_dir: str) -> None:
    print("###################### Train Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("########################################################")

    makedir(exp_dir)
    config_path = os.path.join(exp_dir, "config.yaml")
    resolved_path = os.path.join(exp_dir, "config_resolved.yaml")
    with g_pathmgr.open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    cfg_resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with g_pathmgr.open(resolved_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))


def _resolve_experiment_dir(cfg: DictConfig, args: argparse.Namespace) -> str:
    from datetime import datetime
    
    if args.experiment_dir:
        base_exp_dir = args.experiment_dir
    elif cfg.launcher.experiment_log_dir:
        base_exp_dir = cfg.launcher.experiment_log_dir
    else:
        config_name = Path(args.config).stem
        log_root = Path(args.log_root or DEFAULT_LOG_ROOT)
        base_exp_dir = str(Path(os.getcwd()) / log_root / config_name)
    
    # 현재 날짜와 시간을 YYMMDD_HHMMSS 형태로 생성
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    exp_dir = Path(base_exp_dir) / timestamp
    
    cfg.launcher.experiment_log_dir = str(exp_dir)
    return cfg.launcher.experiment_log_dir


def _setup_submitit_executor(cfg: DictConfig, args: argparse.Namespace, submitit_dir: str):
    executor = submitit.AutoExecutor(folder=submitit_dir)
    submitit_conf = cfg.submitit
    submitit_conf.use_cluster = True

    submitit_conf.partition = (
        args.partition if args.partition is not None else submitit_conf.get("partition")
    )
    submitit_conf.account = (
        args.account if args.account is not None else submitit_conf.get("account")
    )
    submitit_conf.qos = args.qos if args.qos is not None else submitit_conf.get("qos")

    job_kwargs = {
        "timeout_min": 60 * submitit_conf.timeout_hour,
        "name": submitit_conf.name if submitit_conf.get("name") else Path(args.config).stem,
        "slurm_partition": submitit_conf.partition,
        "gpus_per_node": cfg.launcher.gpus_per_node,
        "tasks_per_node": cfg.launcher.gpus_per_node,
        "cpus_per_task": submitit_conf.cpus_per_task,
        "nodes": cfg.launcher.num_nodes,
        "slurm_additional_parameters": {
            "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),
        },
    }

    if "include_nodes" in submitit_conf:
        include_nodes = list(submitit_conf["include_nodes"])
        assert (
            len(include_nodes) >= cfg.launcher.num_nodes
        ), "Not enough nodes in submitit.include_nodes"
        job_kwargs["slurm_additional_parameters"]["nodelist"] = " ".join(include_nodes)

    if submitit_conf.account is not None:
        job_kwargs["slurm_additional_parameters"]["account"] = submitit_conf.account
    if submitit_conf.qos is not None:
        job_kwargs["slurm_additional_parameters"]["qos"] = submitit_conf.qos

    if submitit_conf.get("mem_gb") is not None:
        job_kwargs["mem_gb"] = submitit_conf.mem_gb
    elif submitit_conf.get("mem") is not None:
        job_kwargs["slurm_mem"] = submitit_conf.mem

    if submitit_conf.get("constraints") is not None:
        job_kwargs["slurm_constraint"] = submitit_conf.constraints

    if submitit_conf.get("comment") is not None:
        job_kwargs["slurm_comment"] = submitit_conf.comment

    srun_args = submitit_conf.get("srun_args")
    if srun_args is not None and srun_args.get("cpu_bind") is not None:
        job_kwargs["slurm_srun_args"] = ["--cpu-bind", srun_args.cpu_bind]

    print("###################### SLURM Config ####################")
    print(job_kwargs)
    print("########################################################")
    executor.update_parameters(**job_kwargs)
    return executor


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SAM 2 on SA-V data")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to the fine-tuning config YAML",
    )
    parser.add_argument(
        "--train-frames",
        type=str,
        help="Path to extracted SA-V training frames (JPEGImages_24fps)",
    )
    parser.add_argument(
        "--train-annotations",
        type=str,
        help="Path to SA-V *_manual.json annotations directory",
    )
    parser.add_argument(
        "--file-list",
        type=str,
        help="Optional path to text file listing video ids for training",
    )
    parser.add_argument(
        "--exclude-list",
        type=str,
        help="Optional path to text file listing videos to skip during training",
    )
    parser.add_argument(
        "--val-frames",
        type=str,
        help="Path to extracted SA-V validation frames (JPEGImages_24fps)",
    )
    parser.add_argument(
        "--val-annotations",
        type=str,
        help="Path to SA-V validation annotations directory",
    )
    parser.add_argument(
        "--val-file-list",
        type=str,
        help="Optional path to text file listing validation video ids",
    )
    parser.add_argument(
        "--val-exclude-list",
        type=str,
        help="Optional path to text file listing validation videos to skip",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        help="Destination directory for logs, checkpoints, and configs",
    )
    parser.add_argument(
        "--log-root",
        type=str,
        default=None,
        help="Root directory used when experiment_dir is not provided",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the base SAM 2 checkpoint used for initialization",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs per node",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="Number of nodes to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override number of dataloader workers",
    )
    parser.add_argument(
        "--num-val-workers",
        type=int,
        default=None,
        help="Override number of validation dataloader workers",
    )
    parser.add_argument(
        "--grad-log-every",
        type=int,
        default=None,
        help="Interval (in optimizer steps) for exporting per-layer gradients",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="1 to submit training with Submitit, 0 to force local execution",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")

    args = parser.parse_args()

    register_omegaconf_resolvers()
    cfg: DictConfig = OmegaConf.load(args.config)

    if args.batch_size is not None:
        cfg.scratch.train_batch_size = args.batch_size
    if args.epochs is not None:
        cfg.scratch.num_epochs = args.epochs
    if args.num_workers is not None:
        cfg.scratch.num_train_workers = args.num_workers
    if args.grad_log_every is not None:
        _update_if_not_none(
            cfg,
            "trainer.optim.gradient_logger.log_every",
            args.grad_log_every,
        )
    if args.seed is not None:
        cfg.trainer.seed_value = args.seed

    if args.train_frames is not None:
        cfg.dataset.img_folder = args.train_frames
    if args.train_annotations is not None:
        cfg.dataset.gt_folder = args.train_annotations
    if args.file_list is not None:
        cfg.dataset.file_list_txt = args.file_list
    if args.exclude_list is not None:
        cfg.dataset.excluded_videos_list_txt = args.exclude_list
    if args.val_frames is not None:
        cfg.dataset.val_img_folder = args.val_frames
    if args.val_annotations is not None:
        cfg.dataset.val_gt_folder = args.val_annotations
    if args.val_file_list is not None:
        cfg.dataset.val_file_list_txt = args.val_file_list
    if args.val_exclude_list is not None:
        cfg.dataset.val_excluded_videos_list_txt = args.val_exclude_list
    if args.gpus is not None:
        cfg.launcher.gpus_per_node = args.gpus
    if args.nodes is not None:
        cfg.launcher.num_nodes = args.nodes
    if args.checkpoint is not None:
        _update_if_not_none(
            cfg,
            "trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path",
            args.checkpoint,
        )
    if args.resume is not None:
        _update_if_not_none(cfg, "trainer.checkpoint.resume_from", args.resume)

    prepared_frames_path = None
    if cfg.dataset.img_folder:
        prepared_frames_path = prepare_training_frames_if_needed(cfg.dataset.img_folder)
        if prepared_frames_path is not None:
            if prepared_frames_path != cfg.dataset.img_folder:
                logging.info(
                    "Configured img_folder %s contains videos. Using extracted frames at %s",
                    cfg.dataset.img_folder,
                    prepared_frames_path,
                )
            cfg.dataset.img_folder = prepared_frames_path

    _validate_dataset_paths(cfg)
    validation_enabled = _validate_validation_paths(cfg)
    if validation_enabled:
        _configure_validation(cfg, num_val_workers=args.num_val_workers)
    add_pythonpath_to_sys_path()
    exp_dir = _resolve_experiment_dir(cfg, args)
    _print_and_dump_config(cfg, exp_dir)

    submitit_conf = cfg.submitit
    if submitit_conf is None:
        raise RuntimeError("submitit config missing from YAML")

    if args.use_cluster is not None:
        submitit_conf.use_cluster = bool(args.use_cluster)

    submitit_dir = os.path.join(exp_dir, "submitit_logs")

    if submitit_conf.use_cluster:
        executor = _setup_submitit_executor(cfg, args, submitit_dir)
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        runner = SubmititRunner(main_port, cfg)
        job = executor.submit(runner)
        print(f"Submitit Job ID: {job.job_id}")
        runner.setup_job_info(job.job_id, rank=0)
    else:
        cfg.launcher.num_nodes = 1
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg, main_port)


if __name__ == "__main__":
    main()
