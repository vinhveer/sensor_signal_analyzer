"""ResNet1D with squeeze-excitation blocks and a SwiGLU feed-forward head."""
from __future__ import annotations

import torch
from torch import nn


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.01):
        super().__init__()
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.linear1(x).chunk(2, dim=-1)
        return self.dropout(self.linear2(torch.nn.functional.silu(gate) * value))


class SE1D(nn.Module):
    def __init__(self, channels: int, reduction_ratio: float = 1 / 16):
        super().__init__()
        squeezed_channels = max(1, int(channels * reduction_ratio))
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, squeezed_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(squeezed_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale(x)


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, se_ratio: float):
        super().__init__()
        self.conv_x = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding="same")
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same")
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn_z = nn.BatchNorm1d(out_channels)
        self.se = SE1D(out_channels, se_ratio)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.BatchNorm1d(in_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn_x(self.conv_x(x)))
        out = self.relu(self.bn_y(self.conv_y(out)))
        out = self.se(self.bn_z(self.conv_z(out)))
        return self.relu(out + self.shortcut(x))


class ResNetSESwiGLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        window: int,
        n_feature_maps: int = 64,
        se_ratio: float = 1 / 16,
        ffn_ratio: float = 8 / 3,
        dropout: float = 0.01,
    ):
        super().__init__()
        del window
        n_feature_maps = int(n_feature_maps)
        d_model = n_feature_maps * 2
        self.block1 = ResidualSEBlock(in_channels, n_feature_maps, se_ratio)
        self.block2 = ResidualSEBlock(n_feature_maps, d_model, se_ratio)
        self.block3 = ResidualSEBlock(d_model, d_model, se_ratio)
        self.ffn = SwiGLUFFN(d_model, int(d_model * ffn_ratio), dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.ffn(x.transpose(1, 2)).transpose(1, 2)
        return self.classifier(self.pool(x).squeeze(-1))
