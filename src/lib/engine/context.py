"""Run context: builds the object holding data and run configuration."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from ..data.dataset import DataConfig
from ..data.discovery import prepare_dataset_root


def _build_data_config(config: dict, dataset_root: str) -> DataConfig:
    data = config["data"]
    windowing = config["windowing"]
    split = config["split"]
    return DataConfig(
        root=dataset_root,
        subdir=data["kaggle_dataset_subdir"],
        channels=int(data["expected_channels"]),
        window=int(windowing["window"]),
        step=int(windowing["step_size"]),
        seed=int(split["split_seed"]),
        return_ct=bool(windowing["return_ct"]),
        eps=float(windowing["eps"]),
        expected_files_per_class=int(split["expected_files_per_class"]),
        train_run_ids=list(split["train_run_ids"]),
        val_run_ids=list(split["val_run_ids"]),
        test_run_ids=list(split["test_run_ids"]),
    )


@dataclass
class RunContext:
    global_seed: int
    dataset_name: str
    model_name: str
    dataset_root: str
    classes: list[str]
    cfg: DataConfig
    device: torch.device
    run_dir: str
    version: str
    artifact_prefix: str
    checkpoint_path: str
    n_channels: int
    window: int
    split_seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    grad_clip_norm: float
    use_amp: bool
    model_config: dict


def prepare_run_context(global_seed: int, config: dict, output_root: str) -> RunContext:
    data = config["data"]
    windowing = config["windowing"]
    split = config["split"]
    training = config["training"]
    model_cfg = config["model"]
    dataset_name = config["dataset_name"]
    model_name = config["model_name"]

    os.makedirs(output_root, exist_ok=True)
    dataset_root = prepare_dataset_root(data)
    subdir = Path(data["kaggle_dataset_subdir"])
    classes = sorted(path.name for path in Path(dataset_root).iterdir() if path.is_dir() and (path / subdir).is_dir() and any((path / subdir).glob("*.npy")))
    cfg = _build_data_config(config, dataset_root)

    run_dir = os.path.join(output_root, f"seed{global_seed}")
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    artifact_prefix = f"{dataset_name}_{model_name}"

    return RunContext(
        global_seed=global_seed,
        dataset_name=dataset_name,
        model_name=model_name,
        dataset_root=dataset_root,
        classes=classes,
        cfg=cfg,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        run_dir=run_dir,
        version=f"seed{global_seed}",
        artifact_prefix=artifact_prefix,
        checkpoint_path=os.path.join(run_dir, f"best_{artifact_prefix}.pt"),
        n_channels=int(data["expected_channels"]),
        window=int(windowing["window"]),
        split_seed=int(split["split_seed"]),
        batch_size=int(training["batch_size"]),
        epochs=int(training["epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        num_workers=int(training["num_workers"]),
        grad_clip_norm=float(training["grad_clip_norm"]),
        use_amp=bool(training["use_amp"]),
        model_config=dict(model_cfg),
    )
