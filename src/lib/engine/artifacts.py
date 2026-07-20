"""Shared reporting for every classification model."""
from __future__ import annotations

import os

from ..utils.io import save_json
from ..visualization.confusion_matrix import plot_confusion_matrix
from ..visualization.curves import plot_performance
from .context import RunContext
from .metrics import build_scores


class Reporter:
    def save(self, history: dict, payload: dict, meta: dict, test_true, test_pred, ctx: RunContext) -> dict:
        scores = build_scores(test_true, test_pred, len(ctx.classes))
        run_info = {
            "dataset_name": ctx.dataset_name,
            "model_name": ctx.model_name,
            "global_seed": ctx.global_seed,
            "split_seed": ctx.split_seed,
            "version": ctx.version,
            "device": str(ctx.device),
            "classes": ctx.classes,
            "best_epoch": history["best_epoch"],
            "best_val_acc": history["best_val_acc"],
            "best_val_loss": history["best_val_loss"],
            "test_loss": history["test_loss"],
            "test_acc": history["test_acc"],
            "dataset_root": ctx.dataset_root,
            "config": payload["config"],
            "meta": meta,
        }
        save_json(history, os.path.join(ctx.run_dir, f"history_{ctx.artifact_prefix}.json"))
        save_json(run_info, os.path.join(ctx.run_dir, f"run_info_{ctx.artifact_prefix}.json"))
        save_json(scores, os.path.join(ctx.run_dir, f"scores_{ctx.artifact_prefix}.json"))
        plot_performance(history, os.path.join(ctx.run_dir, f"learning_curve_{ctx.artifact_prefix}"), f"Learning Curve {ctx.dataset_name} {ctx.model_name}")
        plot_confusion_matrix(
            test_true,
            test_pred,
            ctx.classes,
            figure_name=f"Confusion Matrix {ctx.dataset_name} {ctx.model_name} test",
            save_path=os.path.join(ctx.run_dir, f"confusion_matrix_{ctx.artifact_prefix}_test.png"),
        )
        return scores


reporter = Reporter()
