"""Training orchestration: per-seed training and multi-seed runs."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from ..data.dataloaders import make_loaders
from ..utils.archive import zip_run_outputs
from ..utils.io import save_json, save_text
from ..utils.seed import set_global_determinism
from .artifacts import reporter
from .context import RunContext, prepare_run_context


class Trainer:
    @staticmethod
    def synchronize(device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def evaluate(self, model, loader, criterion, ctx):
        model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(ctx.device), yb.to(ctx.device)
                logits = model(xb)
                total_loss += criterion(logits, yb).item() * xb.size(0)
                y_true.append(yb.cpu().numpy())
                y_pred.append(logits.argmax(dim=1).cpu().numpy())
        true = np.concatenate(y_true)
        pred = np.concatenate(y_pred)
        return total_loss / len(loader.dataset), float((true == pred).mean()), true, pred

    def train_epoch(self, model, loader, criterion, optimizer, scaler, ctx):
        model.train()
        total_loss = total_correct = total_seen = 0
        for xb, yb in loader:
            xb = xb.to(ctx.device, non_blocking=True)
            yb = yb.to(ctx.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=ctx.device.type, dtype=torch.float16, enabled=scaler is not None):
                logits = model(xb)
                loss = criterion(logits, yb)
            if scaler is None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ctx.grad_clip_norm)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), ctx.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_seen += xb.size(0)
        return total_loss / total_seen, total_correct / total_seen

    def fit(self, model, train_loader, val_loader, meta, ctx):
        optimizer = torch.optim.AdamW(model.parameters(), lr=ctx.learning_rate, weight_decay=ctx.weight_decay)
        train_criterion = eval_criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
        use_amp = bool(ctx.use_amp and ctx.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
        if ctx.use_amp and not use_amp:
            print(f"AMP requested but unavailable on {ctx.device}; using FP32.")
        if ctx.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(ctx.device)
        history = {
            "loss": [],
            "acc": [],
            "val_loss": [],
            "val_acc": [],
            "train_time_seconds": [],
            "val_time_seconds": [],
            "epoch_time_seconds": [],
            "train_samples_per_second": [],
            "use_amp": use_amp,
        }
        best_val_acc = -1.0
        for epoch in range(1, ctx.epochs + 1):
            self.synchronize(ctx.device)
            epoch_start = time.perf_counter()
            train_start = time.perf_counter()
            train_loss, train_acc = self.train_epoch(model, train_loader, train_criterion, optimizer, scaler, ctx)
            self.synchronize(ctx.device)
            train_time = time.perf_counter() - train_start
            val_start = time.perf_counter()
            val_loss, val_acc, _, _ = self.evaluate(model, val_loader, eval_criterion, ctx)
            self.synchronize(ctx.device)
            val_time = time.perf_counter() - val_start
            epoch_time = time.perf_counter() - epoch_start
            scheduler.step(val_loss)
            history["loss"].append(float(train_loss))
            history["acc"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            history["train_time_seconds"].append(float(train_time))
            history["val_time_seconds"].append(float(val_time))
            history["epoch_time_seconds"].append(float(epoch_time))
            history["train_samples_per_second"].append(float(len(train_loader.dataset) / max(train_time, 1e-12)))
            print(f"Epoch {epoch:03d}/{ctx.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f} | time={epoch_time:.2f}s")
            if val_acc > best_val_acc:
                best_val_acc = float(val_acc)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "best_epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "config": self.checkpoint_config(ctx),
                    "meta": meta,
                    "classes": ctx.classes,
                }, ctx.checkpoint_path)
        return history, eval_criterion

    @staticmethod
    def checkpoint_config(ctx):
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
                "use_amp": ctx.use_amp,
                "global_seed": ctx.global_seed,
                "split_seed": ctx.split_seed,
                "local_centering": True,
                "local_scaling": True,
            },
            "model": ctx.model_config,
        }


trainer = Trainer()


def run_seed(global_seed: int, config: dict, output_root: str, model_factory) -> dict:
    ctx = prepare_run_context(global_seed, config, output_root)
    set_global_determinism(global_seed)
    train_loader, val_loader, test_loader, meta = make_loaders(ctx.classes, ctx.cfg, ctx.batch_size, ctx.num_workers, global_seed)
    model = model_factory(ctx.n_channels, len(ctx.classes), ctx.window, ctx.model_config).to(ctx.device)
    history, eval_criterion = trainer.fit(model, train_loader, val_loader, meta, ctx)
    payload = torch.load(ctx.checkpoint_path, map_location=ctx.device)
    model.load_state_dict(payload["model_state_dict"])
    val_loss, _, _, _ = trainer.evaluate(model, val_loader, eval_criterion, ctx)
    test_loss, test_acc, test_true, test_pred = trainer.evaluate(model, test_loader, eval_criterion, ctx)
    history["best_epoch"] = int(payload["best_epoch"])
    history["best_val_acc"] = float(payload["best_val_acc"])
    history["best_val_loss"] = float(val_loss)
    history["test_loss"] = float(test_loss)
    history["test_acc"] = float(test_acc)
    history["run_dir"] = ctx.run_dir
    history["version"] = ctx.version
    history["peak_gpu_memory_mb"] = float(torch.cuda.max_memory_allocated(ctx.device) / 1024**2) if ctx.device.type == "cuda" else 0.0

    scores = reporter.save(history, payload, meta, test_true, test_pred, ctx)
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


def run_training(config: dict, model_factory) -> dict:
    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    seeds = list(config["training"]["seeds"])
    output_root = config["outputs"]["root"]
    save_zip = bool(config["outputs"].get("save_zip", True))

    os.makedirs(output_root, exist_ok=True)
    results = []
    for seed in seeds:
        print(f"\n===== Training seed {seed}/{seeds[-1]} =====")
        results.append(run_seed(seed, config, output_root, model_factory))

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
