"""ResNet1D inference visualization entry point."""
from __future__ import annotations

from lib.apps import VisualizationApp
from .inference import ResNet1DInferenceApp


class ResNet1DVisualizationApp(VisualizationApp, ResNet1DInferenceApp):
    pass


if __name__ == "__main__":
    ResNet1DVisualizationApp().run()
