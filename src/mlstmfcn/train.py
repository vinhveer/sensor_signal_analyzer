"""MLSTM-FCN training entry point."""
from __future__ import annotations

import argparse

from lib.apps import TrainApp
from .model import MLSTMFCNClassifier


def build_model(in_channels: int, num_classes: int, window: int, config: dict) -> MLSTMFCNClassifier:
    return MLSTMFCNClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        window=window,
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        conv_channels=(config["conv1_channels"], config["conv2_channels"], config["conv3_channels"]),
        lstm_dropout=config["lstm_dropout"],
        conv_dropout=config["conv_dropout"],
        se_reduction=config["se_reduction"],
    )


class MLSTMFCNTrainApp(TrainApp):
    model_name = "mlstmfcn"

    def add_model_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lstm-hidden", type=int, default=128)
        parser.add_argument("--lstm-layers", type=int, default=1)
        parser.add_argument("--conv1-channels", type=int, default=128)
        parser.add_argument("--conv2-channels", type=int, default=256)
        parser.add_argument("--conv3-channels", type=int, default=128)
        parser.add_argument("--lstm-dropout", type=float, default=0.2)
        parser.add_argument("--conv-dropout", type=float, default=0.1)
        parser.add_argument("--se-reduction", type=int, default=16)

    def model_config(self, args: argparse.Namespace) -> dict:
        return {
            "lstm_hidden": args.lstm_hidden,
            "lstm_layers": args.lstm_layers,
            "conv1_channels": args.conv1_channels,
            "conv2_channels": args.conv2_channels,
            "conv3_channels": args.conv3_channels,
            "lstm_dropout": args.lstm_dropout,
            "conv_dropout": args.conv_dropout,
            "se_reduction": args.se_reduction,
        }

    def build_model(self, in_channels: int, num_classes: int, window: int, config: dict) -> MLSTMFCNClassifier:
        return build_model(in_channels, num_classes, window, config)


if __name__ == "__main__":
    MLSTMFCNTrainApp().run()
