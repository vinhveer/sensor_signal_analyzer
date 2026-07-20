"""ResNet-SE-SwiGLU inference entry point."""
from __future__ import annotations

from lib.apps import InferenceApp
from .model import ResNetSESwiGLU
from .train import build_model


class ResNetSESwiGLUInferenceApp(InferenceApp):
    def build_model(self, in_channels: int, num_classes: int, window: int, config: dict) -> ResNetSESwiGLU:
        return build_model(in_channels, num_classes, window, config)


if __name__ == "__main__":
    ResNetSESwiGLUInferenceApp().run()
