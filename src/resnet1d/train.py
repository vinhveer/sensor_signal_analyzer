"""ResNet1D training entry point."""
from __future__ import annotations

import argparse

from lib.apps import TrainApp
from .model import ResNet1D


class ResNet1DTrainApp(TrainApp):
    model_name = "resnet1D"

    def add_model_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--n-feature-maps", type=int, default=64)

    def model_config(self, args: argparse.Namespace) -> dict:
        return {"n_feature_maps": args.n_feature_maps}

    def build_model(self, in_channels: int, num_classes: int, window: int, config: dict) -> ResNet1D:
        return ResNet1D(in_channels, num_classes, window, config["n_feature_maps"])


if __name__ == "__main__":
    ResNet1DTrainApp().run()
