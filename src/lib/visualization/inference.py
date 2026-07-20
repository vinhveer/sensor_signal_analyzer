"""Visualization for inference on one sensor signal."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_inference(signal: np.ndarray, result: dict, save_path: str) -> None:
    classes = result["classes"]
    probabilities = np.asarray([result["probabilities"][name] for name in classes])
    window_probabilities = np.asarray(result["window_probabilities"])
    window_predictions = window_probabilities.argmax(axis=1)
    starts = np.asarray(result["window_starts"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), dpi=180)
    axes[0].plot(signal[:, 0], label="Channel 1", linewidth=1)
    axes[0].plot(signal[:, 1], label="Channel 2", linewidth=1, alpha=0.8)
    axes[0].set_title(f"Signal - predicted: {result['class']}")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    colors = ["#2563eb" if name == result["class"] else "#94a3b8" for name in classes]
    axes[1].bar(classes, probabilities * 100, color=colors)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Probability (%)")
    axes[1].set_title("Mean class probability")
    for index, probability in enumerate(probabilities):
        axes[1].text(index, probability * 100 + 1, f"{probability:.1%}", ha="center")

    axes[2].step(starts, window_predictions, where="post", color="#0f766e")
    axes[2].scatter(starts, window_predictions, c=window_predictions, cmap="tab10", vmin=0, vmax=max(len(classes) - 1, 1))
    axes[2].set_yticks(range(len(classes)), classes)
    axes[2].set_xlabel("Window start")
    axes[2].set_ylabel("Predicted class")
    axes[2].set_title("Prediction by sliding window")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
