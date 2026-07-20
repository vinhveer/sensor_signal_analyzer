"""MLSTM-FCN inference visualization entry point."""
from __future__ import annotations

from lib.apps import VisualizationApp
from .inference import MLSTMFCNInferenceApp


class MLSTMFCNVisualizationApp(VisualizationApp, MLSTMFCNInferenceApp):
    pass


if __name__ == "__main__":
    MLSTMFCNVisualizationApp().run()
