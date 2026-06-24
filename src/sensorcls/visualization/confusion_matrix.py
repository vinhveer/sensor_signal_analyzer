"""Confusion matrix plotting."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, label_list, figure_name="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_list))))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.2%})"

    plt.figure(figsize=(max(6, len(label_list) * 0.7), max(4, len(label_list) * 0.55)))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=label_list,
        yticklabels=label_list,
        cbar=False,
        linewidths=1,
        linecolor="black",
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(figure_name)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
