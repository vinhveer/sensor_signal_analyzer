"""ResNet-SE-SwiGLU inference visualization entry point."""
from __future__ import annotations

from lib.apps import VisualizationApp
from .inference import ResNetSESwiGLUInferenceApp


class ResNetSESwiGLUVisualizationApp(VisualizationApp, ResNetSESwiGLUInferenceApp):
    pass


if __name__ == "__main__":
    ResNetSESwiGLUVisualizationApp().run()
