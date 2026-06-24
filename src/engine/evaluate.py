"""Model evaluation over a DataLoader."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .context import RunContext


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, n_class: int, predict_fn=None):
    predict_fn = predict_fn or (lambda model, xb: model(xb))
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = predict_fn(model, xb)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            total_loss += loss.item() * xb.size(0)
            y_true.append(yb.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    y_probs = np.concatenate(y_probs) if y_probs else np.empty((0, n_class), dtype=np.float32)
    loss = total_loss / max(len(loader.dataset), 1)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    return loss, acc, y_true, y_pred, y_probs


def evaluate_best_model(model: nn.Module, val_loader: DataLoader, test_loader: DataLoader, eval_criterion: nn.Module, ctx: RunContext, predict_fn=None):
    payload = torch.load(ctx.checkpoint_path, map_location=ctx.device)
    model.load_state_dict(payload["model_state_dict"])
    val_loss, _, _, _, _ = evaluate_model(model, val_loader, ctx.device, eval_criterion, len(ctx.classes), predict_fn)
    test_loss, test_acc, test_true, test_pred, _ = evaluate_model(model, test_loader, ctx.device, eval_criterion, len(ctx.classes), predict_fn)
    return payload, val_loss, test_loss, test_acc, test_true, test_pred
