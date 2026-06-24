import torch

from sensorcls.models.cnn1d import Conv1DClassifier


def test_cnn1d_forward_shape():
    model = Conv1DClassifier(
        in_channels=2,
        num_classes=4,
        window=1024,
        fc_dim=128,
        dropout=0.4,
    )
    x = torch.randn(8, 2, 1024)
    y = model(x)
    assert y.shape == (8, 4)
