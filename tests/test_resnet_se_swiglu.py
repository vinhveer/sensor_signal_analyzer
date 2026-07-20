import torch

from resnet_se_swiglu.model import ResNetSESwiGLU
from resnet_se_swiglu.train import ResNetSESwiGLUTrainApp, build_model


def test_resnet_se_swiglu_forward_shape():
    model = ResNetSESwiGLU(
        in_channels=2,
        num_classes=4,
        window=1024,
        n_feature_maps=8,
    )
    assert model(torch.randn(2, 2, 1024)).shape == (2, 4)


def test_resnet_se_swiglu_rebuilds_from_checkpoint_config():
    config = {
        "n_feature_maps": 8,
        "se_ratio": 0.125,
        "ffn_ratio": 2.0,
        "dropout": 0.1,
    }
    model = build_model(2, 4, 1024, config)
    clone = build_model(2, 4, 1024, config)
    clone.load_state_dict(model.state_dict())
    assert clone(torch.randn(2, 2, 1024)).shape == (2, 4)


def test_resnet_se_swiglu_cli_parameters():
    args = ResNetSESwiGLUTrainApp().parser().parse_args([
        "--dataset-root", "data",
        "--amp",
        "--n-feature-maps", "32",
        "--se-ratio", "0.125",
        "--ffn-ratio", "2.0",
        "--dropout", "0.1",
    ])
    assert args.amp is True
    assert args.n_feature_maps == 32
    assert args.se_ratio == 0.125
    assert args.ffn_ratio == 2.0
    assert args.dropout == 0.1
