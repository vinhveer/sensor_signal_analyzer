"""ResNet1D inference entry point."""
from __future__ import annotations

from lib.apps import InferenceApp
from .model import ResNet1D


class ResNet1DInferenceApp(InferenceApp):
    def build_model(self, in_channels: int, num_classes: int, window: int, config: dict) -> ResNet1D:
        return ResNet1D(in_channels, num_classes, window, config["n_feature_maps"])


if __name__ == "__main__":
    ResNet1DInferenceApp().run()
