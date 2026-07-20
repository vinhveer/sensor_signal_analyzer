"""ResNet-SE-SwiGLU training entry point."""
from __future__ import annotations

import argparse

from lib.apps import TrainApp
from .model import ResNetSESwiGLU


def build_model(in_channels: int, num_classes: int, window: int, config: dict) -> ResNetSESwiGLU:
    return ResNetSESwiGLU(
        in_channels=in_channels,
        num_classes=num_classes,
        window=window,
        n_feature_maps=config["n_feature_maps"],
        se_ratio=config["se_ratio"],
        ffn_ratio=config["ffn_ratio"],
        dropout=config["dropout"],
    )


class ResNetSESwiGLUTrainApp(TrainApp):
    model_name = "resnet_se_swiglu"

    def add_model_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--n-feature-maps", type=int, default=64)
        parser.add_argument("--se-ratio", type=float, default=1 / 16)
        parser.add_argument("--ffn-ratio", type=float, default=8 / 3)
        parser.add_argument("--dropout", type=float, default=0.01)

    def model_config(self, args: argparse.Namespace) -> dict:
        return {
            "n_feature_maps": args.n_feature_maps,
            "se_ratio": args.se_ratio,
            "ffn_ratio": args.ffn_ratio,
            "dropout": args.dropout,
        }

    def build_model(self, in_channels: int, num_classes: int, window: int, config: dict) -> ResNetSESwiGLU:
        return build_model(in_channels, num_classes, window, config)


if __name__ == "__main__":
    ResNetSESwiGLUTrainApp().run()
