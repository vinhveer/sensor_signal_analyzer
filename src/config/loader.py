"""Default configuration and validation."""
from __future__ import annotations

from copy import deepcopy

REQUIRED_TOP_LEVEL_KEYS = ["dataset_name", "model_name", "data", "windowing", "split", "training", "model", "outputs"]

DEFAULT_CONFIG = {
    "dataset_name": "dataset_1_10",
    "model_name": "1dcnn",
    "data": {
        "kaggle_dataset_subdir": "Processed",
        "kaggle_working_dataset_root": "/kaggle/working/dataset_1_10",
        "kaggle_input_root_candidates": [
            "/kaggle/input/datasets/thanhhieu03092004/data-set-1-10/dataset_1_10",
        ],
        "expected_channels": 2,
        "end_t": None,
        "convert_overwrite": False,
        "acc_columns": ["Acceleration - x (m/s\u00b2)", "Acceleration - x (m/s\u00b2).1"],
        "source_subdir_candidates": [".", "Processed"],
        "csv_sep_candidates": [",", ";", None],
    },
    "windowing": {"window": 1024, "overlap": 0.75, "return_ct": True, "eps": 1e-8},
    "split": {
        "expected_files_per_class": 18,
        "train_run_ids": [1, 2, 3, 4],
        "val_run_ids": [5],
        "test_run_ids": [6],
        "split_seed": 42,
    },
    "training": {
        "seeds": [42],
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "num_workers": 0,
        "grad_clip_norm": 1.0,
    },
    "model": {"name": "cnn1d", "fc_dim": 128, "dropout": 0.0},
    "outputs": {"root": "/kaggle/working/History", "save_zip": True},
}


def default_config() -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    finalize_config(config)
    return config


def finalize_config(config: dict) -> None:
    _validate_config(config)
    _compute_derived_fields(config)


def _validate_config(config: dict) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    windowing = config["windowing"]
    for key in ("window", "overlap"):
        if key not in windowing:
            raise ValueError(f"Config 'windowing' missing required key: {key}")
    data = config["data"]
    if "expected_channels" not in data:
        raise ValueError("Config 'data' missing required key: expected_channels")


def _compute_derived_fields(config: dict) -> None:
    windowing = config["windowing"]
    window = int(windowing["window"])
    overlap = float(windowing["overlap"])
    if window <= 0:
        raise ValueError("Config 'windowing.window' must be > 0")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Config 'windowing.overlap' must be >= 0 and < 1")
    if "step_size" not in windowing or windowing.get("step_size") is None:
        windowing["step_size"] = int(window * (1.0 - overlap))
    if int(windowing["step_size"]) <= 0:
        raise ValueError("Config 'windowing.step_size' must be > 0")
    windowing.setdefault("eps", 1e-8)
    windowing.setdefault("return_ct", True)
