"""Checkpoint payload construction and saving."""
from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from .context import RunContext


def _checkpoint_config(ctx: RunContext) -> dict:
    return {
        "dataset_name": ctx.dataset_name,
        "model_name": ctx.model_name,
        "global_seed": ctx.global_seed,
        "data": asdict(ctx.cfg),
        "training": {
            "batch_size": ctx.batch_size,
            "epochs": ctx.epochs,
            "learning_rate": ctx.learning_rate,
            "weight_decay": ctx.weight_decay,
            "num_workers": ctx.num_workers,
            "grad_clip_norm": ctx.grad_clip_norm,
            "global_seed": ctx.global_seed,
            "split_seed": ctx.split_seed,
            "local_centering": True,
            "local_scaling": True,
        },
        "model": ctx.model_config,
    }


def save_checkpoint(model: nn.Module, optimizer, history: dict, best_epoch: int, best_val_acc: float, meta: dict, ctx: RunContext) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "config": _checkpoint_config(ctx),
        "meta": meta,
        "classes": ctx.classes,
    }, ctx.checkpoint_path)
