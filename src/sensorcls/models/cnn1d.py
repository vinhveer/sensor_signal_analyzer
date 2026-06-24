"""1D CNN classifier with TensorFlow-style 'same' padding convolutions."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Conv1dSame(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length_in = x.shape[-1]
        length_out = math.ceil(length_in / self.stride)
        pad_needed = max(0, (length_out - 1) * self.stride + self.kernel_size - length_in)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        return self.conv(F.pad(x, (pad_left, pad_right)))

class Conv1DClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, window: int, fc_dim: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            Conv1dSame(in_channels, 8, 51, 2), nn.BatchNorm1d(8), nn.ReLU(), nn.MaxPool1d(2),
            Conv1dSame(8, 16, 21, 2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            Conv1dSame(16, 32, 11, 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            Conv1dSame(32, 64, 7, 2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, in_channels, window)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
