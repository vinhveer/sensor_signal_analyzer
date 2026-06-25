import torch
import pytest

from models import build_model
from models.resnet1D import ResNet1D


def test_resnet1d_forward_shape():
    model = build_model(
        {"name": "resnet1D", "n_feature_maps": 64},
        in_channels=2,
        num_classes=4,
        window=1024,
    )
    x = torch.randn(8, 2, 1024)
    y = model(x)
    assert y.shape == (8, 4)
    assert isinstance(model, ResNet1D)


def test_build_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model({"name": "missing"}, in_channels=2, num_classes=4, window=1024)


def test_resnet1d_rejects_unknown_params():
    with pytest.raises(ValueError, match="Unknown resnet1D params"):
        build_model({"name": "resnet1D", "unknown": 1}, in_channels=2, num_classes=4, window=1024)
