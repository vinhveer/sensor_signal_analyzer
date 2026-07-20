"""MLSTM-FCN classifier with squeeze-excitation blocks."""
from __future__ import annotations

import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.scale = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale(self.pool(x).squeeze(-1)).unsqueeze(-1)
        return x * scale


class MLSTMFCNClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        window: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        conv_channels: tuple[int, int, int] = (128, 256, 128),
        lstm_dropout: float = 0.2,
        conv_dropout: float = 0.1,
        se_reduction: int = 16,
    ):
        super().__init__()
        del window
        conv1, conv2, conv3 = conv_channels
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.conv1 = nn.Conv1d(in_channels, conv1, kernel_size=8)
        self.conv2 = nn.Conv1d(conv1, conv2, kernel_size=5)
        self.conv3 = nn.Conv1d(conv2, conv3, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(conv1)
        self.bn2 = nn.BatchNorm1d(conv2)
        self.bn3 = nn.BatchNorm1d(conv3)
        self.se1 = SELayer(conv1, se_reduction)
        self.se2 = SELayer(conv2, se_reduction)
        self.se3 = SELayer(conv3, se_reduction)
        self.conv_dropout = nn.Dropout(conv_dropout)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(conv3 + lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_output, _ = self.lstm(x.transpose(1, 2))
        lstm_features = self.lstm_dropout(lstm_output[:, -1, :])

        conv = self.conv_dropout(self.relu(self.bn1(self.conv1(x))))
        conv = self.se1(conv)
        conv = self.conv_dropout(self.relu(self.bn2(self.conv2(conv))))
        conv = self.se2(conv)
        conv = self.conv_dropout(self.relu(self.bn3(self.conv3(conv))))
        conv_features = self.se3(conv).mean(dim=2)

        return self.classifier(torch.cat((lstm_features, conv_features), dim=1))
