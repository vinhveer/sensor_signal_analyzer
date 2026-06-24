"""Construction of train/val/test DataLoaders and split metadata."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..utils.seed import seed_worker
from .dataset import DataConfig, WindowDataset, build_index, list_npy_files, split_files_for_class


def make_loaders(classes: list[str], cfg: DataConfig, batch_size: int, num_workers: int, loader_seed: int):
    train_files, val_files, test_files = [], [], []
    per_class = {}
    for label, class_name in enumerate(classes):
        files = list_npy_files(cfg, class_name)
        train_split, val_split, test_split = split_files_for_class(files, cfg)
        train_files += [(path, label) for path in train_split]
        val_files += [(path, label) for path in val_split]
        test_files += [(path, label) for path in test_split]
        per_class[class_name] = {"files": len(files), "train_files": len(train_split), "val_files": len(val_split), "test_files": len(test_split)}
    train_index = build_index(train_files, cfg)
    val_index = build_index(val_files, cfg)
    test_index = build_index(test_files, cfg)
    train_ds = WindowDataset(train_index, cfg)
    val_ds = WindowDataset(val_index, cfg)
    test_ds = WindowDataset(test_index, cfg)
    generator = torch.Generator().manual_seed(int(loader_seed))
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker if num_workers > 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    meta = {
        "split_files": {
            "train": [{"file": path, "label": int(label), "class": classes[int(label)]} for path, label in train_files],
            "val": [{"file": path, "label": int(label), "class": classes[int(label)]} for path, label in val_files],
            "test": [{"file": path, "label": int(label), "class": classes[int(label)]} for path, label in test_files],
        },
        "window_stride": int(cfg.step),
        "window_overlap": float(1.0 - cfg.step / cfg.window),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "test_files": len(test_files),
        "train_samples": len(train_index),
        "val_samples": len(val_index),
        "test_samples": len(test_index),
        "per_class": per_class,
    }
    return train_loader, val_loader, test_loader, meta
