"""Training orchestration: per-seed training and multi-seed runs."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.dataloaders import make_loaders
from models import build_model, build_training_objects, predict_logits
from utils.archive import zip_run_outputs
from utils.io import save_json, save_text
from utils.seed import set_global_determinism
from .artifacts import build_run_info, save_run_artifacts
from .checkpoint import save_checkpoint
from .context import RunContext, prepare_run_context
from .evaluate import evaluate_best_model, evaluate_model
from .metrics import build_scores

def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, optimizer, grad_clip_norm: float, predict_fn) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = predict_fn(model, xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * xb.size(0)
        total_correct += (preds == yb).sum().item()
        total_seen += xb.size(0)
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def train_epochs(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer, scheduler, train_criterion: nn.Module, eval_criterion: nn.Module, meta: dict, ctx: RunContext, predict_fn) -> dict:
    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, ctx.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, ctx.device, train_criterion, optimizer, ctx.grad_clip_norm, predict_fn)
        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, ctx.device, eval_criterion, len(ctx.classes), predict_fn)
        scheduler.step(val_loss)
        history["loss"].append(float(train_loss))
        history["acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        print(f"Epoch {epoch:03d}/{ctx.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_epoch = int(epoch)
            save_checkpoint(model, optimizer, history, best_epoch, best_val_acc, meta, ctx)
    return history


def run_seed(global_seed: int, config: dict, output_root: str) -> dict:
    ctx = prepare_run_context(global_seed, config, output_root)
    set_global_determinism(global_seed)
    train_loader, val_loader, test_loader, meta = make_loaders(ctx.classes, ctx.cfg, ctx.batch_size, ctx.num_workers, global_seed)
    model = build_model(ctx.model_config, in_channels=ctx.n_channels, num_classes=len(ctx.classes), window=ctx.window).to(ctx.device)
    model_predict = lambda model, xb: predict_logits(ctx.model_config, model, xb)
    optimizer, train_criterion, eval_criterion, scheduler = build_training_objects(ctx.model_config, model, ctx)

    history = train_epochs(model, train_loader, val_loader, optimizer, scheduler, train_criterion, eval_criterion, meta, ctx, model_predict)

    payload, val_loss, test_loss, test_acc, test_true, test_pred = evaluate_best_model(model, val_loader, test_loader, eval_criterion, ctx, model_predict)
    history["best_epoch"] = int(payload["best_epoch"])
    history["best_val_acc"] = float(payload["best_val_acc"])
    history["best_val_loss"] = float(val_loss)
    history["test_loss"] = float(test_loss)
    history["test_acc"] = float(test_acc)
    history["run_dir"] = ctx.run_dir
    history["version"] = ctx.version

    scores = build_scores(test_true, test_pred, len(ctx.classes))
    run_info = build_run_info(history, payload, meta, ctx)
    save_run_artifacts(history, run_info, scores, test_true, test_pred, ctx)
    print(json.dumps({"run_dir": ctx.run_dir, "scores": scores}, indent=2))
    return {"seed": global_seed, "global_seed": global_seed, "split_seed": ctx.split_seed, "run_dir": ctx.run_dir, "scores": scores}


def write_seed_stats(results: list[dict], output_dir: str, dataset_name: str, model_name: str, seeds: list[int]) -> str:
    metrics = ["accuracy", "precision", "recall", "f1score"]
    seed_label = "_".join(str(seed) for seed in seeds)
    stats_path = os.path.join(output_dir, f"seed_stats_{dataset_name}_{model_name}_seeds{seed_label}.txt")
    lines = [
        f"Dataset: {dataset_name}",
        f"Model: {model_name}",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "Per-seed scores:",
    ]
    for item in results:
        scores = item.get("scores", {})
        parts = [f"seed{item.get('seed')}"]
        for metric in metrics:
            if metric in scores:
                parts.append(f"{metric}={float(scores[metric]):.6f}")
        lines.append("  " + ", ".join(parts))

    lines.extend(["", "Mean +- std:"])
    for metric in metrics:
        values = np.asarray([float(item["scores"][metric]) for item in results if metric in item.get("scores", {})], dtype=np.float64)
        if values.size == 0:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        lines.append(f"  {metric}: {mean:.6f} +- {std:.6f}")

    save_text(lines, stats_path)
    return stats_path


def run_training(config: dict) -> dict:
    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    seeds = list(config["training"]["seeds"])
    output_root = config["outputs"]["root"]
    save_zip = bool(config["outputs"].get("save_zip", True))

    os.makedirs(output_root, exist_ok=True)
    results = []
    for seed in seeds:
        print(f"\n===== Training seed {seed}/{seeds[-1]} =====")
        results.append(run_seed(seed, config, output_root))

    stats_path = write_seed_stats(results, output_root, dataset_name, model_name, seeds)
    summary = {"stats_path": stats_path, "results": results}
    if save_zip:
        zip_path = zip_run_outputs(output_root, dataset_name, model_name, seeds, stats_path)
        summary["zip_path"] = zip_path
    seed_label = "_".join(str(seed) for seed in seeds)
    summary_path = os.path.join(output_root, f"summary_{dataset_name}_{model_name}_seeds{seed_label}.json")
    save_json(summary, summary_path)
    summary["summary_path"] = summary_path
    summary["output_root"] = output_root
    print(json.dumps(summary, indent=2))
    return summary
