from __future__ import annotations

import torch
from torch import nn

from .model import Conv1DClassifier
from .params import Conv1DParams


def build_model(model_config: dict, in_channels: int, num_classes: int, window: int) -> Conv1DClassifier:
    params = Conv1DParams.from_config(model_config)
    return Conv1DClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        window=window,
        fc_dim=params.fc_dim,
        dropout=params.dropout,
    )


def build_training_objects(model: nn.Module, ctx):
    optimizer = torch.optim.AdamW(model.parameters(), lr=ctx.learning_rate, weight_decay=ctx.weight_decay)
    train_criterion = nn.CrossEntropyLoss()
    eval_criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    return optimizer, train_criterion, eval_criterion, scheduler
