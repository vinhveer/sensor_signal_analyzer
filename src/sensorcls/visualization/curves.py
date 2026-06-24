"""Learning curve plotting."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_performance(history: dict, save_base: str, title: str) -> None:
    xs = np.arange(1, len(history["loss"]) + 1)
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    ax1.plot(xs, np.asarray(history["acc"]) * 100.0, color="red", label="acc")
    ax1.plot(xs, np.asarray(history["val_acc"]) * 100.0, color="blue", label="val_acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(title)
    ax1.grid(axis="x", color="0.85")
    ax1.grid(axis="y", color="0.85")
    ax1.legend(loc="best")
    ax2.plot(xs, history["loss"], color="red", label="loss")
    ax2.plot(xs, history["val_loss"], color="blue", label="val_loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(axis="x", color="0.85")
    ax2.grid(axis="y", color="0.85")
    ax2.legend(loc="best")
    fig.tight_layout(pad=0.8)
    fig.savefig(save_base + ".png", dpi=500, bbox_inches="tight")
    plt.close(fig)
