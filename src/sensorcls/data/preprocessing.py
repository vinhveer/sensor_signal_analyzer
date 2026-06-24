"""Sliding window position computation and per-window normalization."""
from __future__ import annotations

import numpy as np


def starts_for_length(length: int, window: int, step: int) -> list[int]:
    if length < window:
        return []
    return list(range(0, length - window + 1, step))


def preprocess_window(window: np.ndarray, eps: float) -> np.ndarray:
    processed = np.asarray(window, dtype=np.float32)
    processed = processed - processed.mean(axis=0, keepdims=True)
    local_std = processed.std(axis=0, keepdims=True)
    processed = processed / (local_std + eps)
    return processed
