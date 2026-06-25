"""Command line entry point for training."""
from __future__ import annotations

import argparse

from config.loader import default_config, finalize_config
from engine.train import run_training
from models import add_model_cli_args, apply_model_cli_overrides


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    data_config = config["data"]
    if args.dataset_root is not None:
        data_config["kaggle_working_dataset_root"] = args.dataset_root
    elif args.kaggle_working_dataset_root is not None:
        data_config["kaggle_working_dataset_root"] = args.kaggle_working_dataset_root
    if args.kaggle_input_root is not None:
        data_config["kaggle_input_root_candidates"] = args.kaggle_input_root
    if args.output_root is not None:
        config["outputs"]["root"] = args.output_root
    if args.no_save_zip:
        config["outputs"]["save_zip"] = False
    if args.window is not None:
        config["windowing"]["window"] = args.window
        config["windowing"]["step_size"] = None
    if args.overlap is not None:
        config["windowing"]["overlap"] = args.overlap
        config["windowing"]["step_size"] = None
    if args.train_run_ids is not None:
        config["split"]["train_run_ids"] = parse_int_list(args.train_run_ids)
    if args.val_run_ids is not None:
        config["split"]["val_run_ids"] = parse_int_list(args.val_run_ids)
    if args.test_run_ids is not None:
        config["split"]["test_run_ids"] = parse_int_list(args.test_run_ids)
    if args.expected_files_per_class is not None:
        config["split"]["expected_files_per_class"] = args.expected_files_per_class
    if args.seeds is not None:
        config["training"]["seeds"] = parse_int_list(args.seeds)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.grad_clip_norm is not None:
        config["training"]["grad_clip_norm"] = args.grad_clip_norm
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    apply_model_cli_overrides(config["model"], args)
    finalize_config(config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sensor time-series classifier.")
    parser.add_argument("--dataset-root", help="Dataset root containing class folders. Alias for --kaggle-working-dataset-root.")
    parser.add_argument(
        "--kaggle-working-dataset-root",
        help="Writable dataset root used after CSV-to-NPY conversion, for example /kaggle/working/dataset_1_10.",
    )
    parser.add_argument(
        "--kaggle-input-root",
        action="append",
        help="Kaggle input dataset root candidate. Can be passed multiple times.",
    )
    parser.add_argument("--output-root", help="Output directory for run artifacts.")
    parser.add_argument("--no-save-zip", action="store_true", help="Disable output zip creation.")
    parser.add_argument("--window", type=int, help="Sliding window length.")
    parser.add_argument("--overlap", type=float, help="Sliding window overlap, for example 0.75.")
    parser.add_argument("--train-run-ids", help="Comma-separated train run ids, for example 1,2,3,4.")
    parser.add_argument("--val-run-ids", help="Comma-separated validation run ids, for example 5.")
    parser.add_argument("--test-run-ids", help="Comma-separated test run ids, for example 6.")
    parser.add_argument("--expected-files-per-class", type=int, help="Expected .npy files per class.")
    parser.add_argument("--seeds", help="Comma-separated training seeds, for example 42,43,44.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--epochs", type=int, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Optimizer weight decay.")
    parser.add_argument("--num-workers", type=int, help="DataLoader worker count.")
    parser.add_argument("--grad-clip-norm", type=float, help="Gradient clipping norm.")
    parser.add_argument("--model-name", help="Override model.name from config, for example resnet1D.")
    add_model_cli_args(parser)
    args = parser.parse_args()

    config = default_config()
    config = apply_cli_overrides(config, args)
    summary = run_training(config)
    print(f"Outputs written to: {summary.get('output_root')}")


if __name__ == "__main__":
    main()
