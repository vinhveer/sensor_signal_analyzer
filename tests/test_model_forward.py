import torch

from resnet1d.model import ResNet1D


def test_resnet1d_forward_shape():
    model = ResNet1D(in_channels=2, num_classes=4, window=1024, n_feature_maps=64)
    x = torch.randn(8, 2, 1024)
    y = model(x)
    assert y.shape == (8, 4)
    assert isinstance(model, ResNet1D)
