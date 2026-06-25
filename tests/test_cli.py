from types import SimpleNamespace

from cli.train import apply_cli_overrides, parse_int_list
from config.loader import default_config


def args(**overrides):
    values = {
        "kaggle_working_dataset_root": None,
        "dataset_root": None,
        "kaggle_input_root": None,
        "output_root": None,
        "no_save_zip": False,
        "window": None,
        "overlap": None,
        "train_run_ids": None,
        "val_run_ids": None,
        "test_run_ids": None,
        "expected_files_per_class": None,
        "seeds": None,
        "batch_size": None,
        "epochs": None,
        "learning_rate": None,
        "weight_decay": None,
        "num_workers": None,
        "grad_clip_norm": None,
        "model_name": None,
        "n_feature_maps": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_model_cli_overrides_merge_into_config():
    config = default_config()
    apply_cli_overrides(config, args(model_name="resnet1D", n_feature_maps=32))
    assert config["model"] == {"name": "resnet1D", "n_feature_maps": 32}


def test_training_cli_overrides_merge_into_config():
    config = default_config()
    apply_cli_overrides(config, args(dataset_root="D:/data", output_root="D:/out", epochs=2, batch_size=8, seeds="1,2", window=512, overlap=0.5, no_save_zip=True))
    assert config["data"]["kaggle_working_dataset_root"] == "D:/data"
    assert config["outputs"] == {"root": "D:/out", "save_zip": False}
    assert config["training"]["epochs"] == 2
    assert config["training"]["batch_size"] == 8
    assert config["training"]["seeds"] == [1, 2]
    assert config["windowing"]["step_size"] == 256


def test_parse_int_list():
    assert parse_int_list("1, 2,3") == [1, 2, 3]
