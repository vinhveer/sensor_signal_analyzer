"""Command line entry point for training."""
from __future__ import annotations

import argparse

from ..config.loader import load_config
from ..engine.train import run_training


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    data_config = config["data"]
    if args.kaggle_working_dataset_root is not None:
        data_config["kaggle_working_dataset_root"] = args.kaggle_working_dataset_root
    if args.kaggle_input_root is not None:
        data_config["kaggle_input_root_candidates"] = args.kaggle_input_root
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sensor time-series classifier from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--kaggle-working-dataset-root",
        help="Writable dataset root used after CSV-to-NPY conversion, for example /kaggle/working/dataset_1_10.",
    )
    parser.add_argument(
        "--kaggle-input-root",
        action="append",
        help="Kaggle input dataset root candidate. Can be passed multiple times.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    summary = run_training(config)
    print(f"Outputs written to: {summary.get('output_root')}")


if __name__ == "__main__":
    main()
