"""Classification metric aggregation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def build_scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> dict:
    labels = None if n_classes is None else list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
