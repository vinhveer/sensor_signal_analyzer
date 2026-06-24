from __future__ import annotations

from . import cnn1d


MODEL_REGISTRY = {
    "cnn1d": cnn1d,
}


def get_model_module(model_config: dict):
    name = model_config["name"]
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}") from exc


def build_model(model_config: dict, in_channels: int, num_classes: int, window: int):
    return get_model_module(model_config).build_model(model_config, in_channels, num_classes, window)


def add_model_cli_args(parser) -> None:
    for module in MODEL_REGISTRY.values():
        if hasattr(module, "add_cli_args"):
            module.add_cli_args(parser)


def apply_model_cli_overrides(model_config: dict, args) -> None:
    module = get_model_module(model_config)
    if hasattr(module, "apply_cli_overrides"):
        module.apply_cli_overrides(model_config, args)


def build_training_objects(model_config: dict, model, ctx):
    return get_model_module(model_config).build_training_objects(model, ctx)


def predict_logits(model_config: dict, model, xb):
    return get_model_module(model_config).predict_logits(model, xb)
