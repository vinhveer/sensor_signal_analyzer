"""MLSTM-FCN inference entry point."""
from __future__ import annotations

from lib.apps import InferenceApp
from .model import MLSTMFCNClassifier
from .train import build_model


class MLSTMFCNInferenceApp(InferenceApp):
    def build_model(self, in_channels: int, num_classes: int, window: int, config: dict) -> MLSTMFCNClassifier:
        return build_model(in_channels, num_classes, window, config)


if __name__ == "__main__":
    MLSTMFCNInferenceApp().run()
