"""YAML configuration loading and validation."""
from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_TOP_LEVEL_KEYS = ["dataset_name", "model_name", "data", "windowing", "split", "training", "model", "outputs"]


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping, got: {type(config)}")
    _validate_config(config)
    _compute_derived_fields(config)
    return config


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
    if "step_size" not in windowing or windowing.get("step_size") is None:
        windowing["step_size"] = int(window * (1.0 - overlap))
    windowing.setdefault("eps", 1e-8)
    windowing.setdefault("return_ct", True)
