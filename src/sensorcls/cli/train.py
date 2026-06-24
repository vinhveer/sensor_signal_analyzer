"""Command line entry point for training."""
from __future__ import annotations

import argparse

from ..config.loader import load_config
from ..engine.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sensor time-series classifier from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = run_training(config)
    print(f"Outputs written to: {summary.get('output_root')}")


if __name__ == "__main__":
    main()
