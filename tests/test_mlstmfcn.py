import torch
import numpy as np

from mlstmfcn.visualize import MLSTMFCNVisualizationApp
from mlstmfcn.model import MLSTMFCNClassifier
from mlstmfcn.train import MLSTMFCNTrainApp, build_model


def test_mlstmfcn_forward_shape():
    model = MLSTMFCNClassifier(
        in_channels=2,
        num_classes=4,
        window=1024,
        lstm_hidden=8,
        conv_channels=(8, 16, 8),
    )
    assert model(torch.randn(2, 2, 1024)).shape == (2, 4)


def test_mlstmfcn_rebuilds_from_checkpoint_config():
    config = {
        "lstm_hidden": 8,
        "lstm_layers": 1,
        "conv1_channels": 8,
        "conv2_channels": 16,
        "conv3_channels": 8,
        "lstm_dropout": 0.2,
        "conv_dropout": 0.1,
        "se_reduction": 4,
    }
    model = build_model(2, 4, 1024, config)
    clone = build_model(2, 4, 1024, config)
    clone.load_state_dict(model.state_dict())
    assert clone(torch.randn(2, 2, 1024)).shape == (2, 4)


def test_mlstmfcn_cli_parameters():
    args = MLSTMFCNTrainApp().parser().parse_args([
        "--dataset-root", "data",
        "--lstm-hidden", "64",
        "--conv1-channels", "32",
        "--se-reduction", "8",
    ])
    assert args.lstm_hidden == 64
    assert args.conv1_channels == 32
    assert args.se_reduction == 8


def test_visualization_creates_json_and_png(tmp_path):
    config = {
        "name": "mlstmfcn",
        "lstm_hidden": 8,
        "lstm_layers": 1,
        "conv1_channels": 8,
        "conv2_channels": 16,
        "conv3_channels": 8,
        "lstm_dropout": 0.2,
        "conv_dropout": 0.1,
        "se_reduction": 4,
    }
    model = build_model(2, 4, 1024, config)
    checkpoint = tmp_path / "model.pt"
    signal_path = tmp_path / "signal.npy"
    output_dir = tmp_path / "visualization"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": ["Dinh", "Giang", "Healthy", "Nut"],
        "config": {
            "data": {"channels": 2, "window": 1024, "step": 256},
            "model": config,
        },
    }, checkpoint)
    np.save(signal_path, np.random.default_rng(42).normal(size=(2048, 2)).astype(np.float32))

    result = MLSTMFCNVisualizationApp().run([
        "--checkpoint", str(checkpoint),
        "--input", str(signal_path),
        "--output-dir", str(output_dir),
    ])

    assert result["class"] in ["Dinh", "Giang", "Healthy", "Nut"]
    assert (output_dir / "inference_signal.json").is_file()
    assert (output_dir / "inference_signal.png").is_file()
