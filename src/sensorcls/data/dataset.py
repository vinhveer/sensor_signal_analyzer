"""PyTorch Dataset, data configuration, and run-id based file splitting."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_window, starts_for_length


@dataclass
class DataConfig:
    root: str
    subdir: str
    channels: int
    window: int
    step: int
    seed: int
    return_ct: bool
    eps: float = 1e-8
    expected_files_per_class: int = 18
    train_run_ids: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    val_run_ids: list[int] = field(default_factory=lambda: [5])
    test_run_ids: list[int] = field(default_factory=lambda: [6])


def class_dir(cfg: DataConfig, class_name: str) -> Path:
    return Path(cfg.root) / class_name / cfg.subdir


def list_npy_files(cfg: DataConfig, class_name: str) -> list[str]:
    directory = class_dir(cfg, class_name)
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    files = sorted(str(path) for path in directory.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {directory}")
    return files


def load_npy_mmap(path: str, cache: dict[str, np.ndarray], cfg: DataConfig) -> np.ndarray:
    if path not in cache:
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or int(array.shape[1]) != int(cfg.channels):
            raise ValueError(f"Invalid array shape in {path}: {array.shape}")
        cache[path] = array
    return cache[path]


def load_window(path: str, start: int, cfg: DataConfig, cache: dict[str, np.ndarray]) -> np.ndarray:
    window = np.asarray(load_npy_mmap(path, cache, cfg)[start:start + cfg.window, :], dtype=np.float32)
    if int(window.shape[0]) != int(cfg.window):
        raise ValueError(f"Window length mismatch in {path}: {window.shape[0]} != {cfg.window}")
    return window


def build_index(files_with_label: list[tuple[str, int]], cfg: DataConfig) -> list[tuple[str, int, int]]:
    cache: dict[str, np.ndarray] = {}
    index = []
    for file_path, label in files_with_label:
        array = load_npy_mmap(file_path, cache, cfg)
        for start in starts_for_length(int(array.shape[0]), cfg.window, cfg.step):
            index.append((file_path, label, int(start)))
    return index


class WindowDataset(Dataset):
    def __init__(self, index: list[tuple[str, int, int]], cfg: DataConfig):
        self.index = index
        self.cfg = cfg
        self.cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        file_path, label, start = self.index[idx]
        window = preprocess_window(load_window(file_path, start, self.cfg, self.cache), self.cfg.eps)
        if self.cfg.return_ct:
            window = window.T
        return torch.from_numpy(window), torch.tensor(label, dtype=torch.long)


def run_id_from_filename(file_path: str) -> int:
    run_match = re.search(r"R(\d+)", Path(file_path).stem)
    if run_match is None:
        raise ValueError(f"Cannot parse run id from filename: {file_path}")
    return int(run_match.group(1))


def split_files_for_class(files: list[str], cfg: DataConfig) -> tuple[list[str], list[str], list[str]]:
    if len(files) != cfg.expected_files_per_class:
        raise ValueError(f"Expected {cfg.expected_files_per_class} files per class, got n={len(files)}")
    train_files = []
    val_files = []
    test_files = []

    for file_path in files:
        run_id = run_id_from_filename(file_path)
        if run_id in cfg.train_run_ids:
            train_files.append(file_path)
        elif run_id in cfg.val_run_ids:
            val_files.append(file_path)
        elif run_id in cfg.test_run_ids:
            test_files.append(file_path)
        else:
            raise ValueError(f"Unexpected run id R{run_id} in file: {file_path}")

    return sorted(train_files), sorted(val_files), sorted(test_files)
