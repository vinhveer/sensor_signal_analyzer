from lib.apps import parse_int_list
from resnet1d.train import ResNet1DTrainApp


def test_parse_int_list():
    assert parse_int_list("1, 2,3") == [1, 2, 3]


def test_resnet_train_cli_has_model_parameters():
    args = ResNet1DTrainApp().parser().parse_args(["--dataset-root", "data", "--n-feature-maps", "32"])
    assert args.n_feature_maps == 32
