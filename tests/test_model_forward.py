import torch
import pytest

from models import build_model
from models.cnn1d import Conv1DClassifier


def test_cnn1d_forward_shape():
    model = build_model(
        {"name": "cnn1d", "fc_dim": 128, "dropout": 0.4},
        in_channels=2,
        num_classes=4,
        window=1024,
    )
    x = torch.randn(8, 2, 1024)
    y = model(x)
    assert y.shape == (8, 4)
    assert isinstance(model, Conv1DClassifier)


def test_build_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"name": "missing"}, in_channels=2, num_classes=4, window=1024)


def test_cnn1d_rejects_unknown_params():
    with pytest.raises(ValueError, match="Unknown cnn1d params"):
        build_model({"name": "cnn1d", "unknown": 1}, in_channels=2, num_classes=4, window=1024)
