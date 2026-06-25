from __future__ import annotations

import torch
from torch import nn

from .model import ResNet1D
from .params import ResNet1DParams


def build_model(model_config: dict, in_channels: int, num_classes: int, _window: int) -> ResNet1D:
    params = ResNet1DParams.from_config(model_config)
    return ResNet1D(
        in_channels=in_channels,
        n_feature_maps=params.n_feature_maps,
        num_classes=num_classes,
    )


def build_training_objects(model: nn.Module, ctx):
    optimizer = torch.optim.AdamW(model.parameters(), lr=ctx.learning_rate, weight_decay=ctx.weight_decay)
    train_criterion = nn.CrossEntropyLoss()
    eval_criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    return optimizer, train_criterion, eval_criterion, scheduler
