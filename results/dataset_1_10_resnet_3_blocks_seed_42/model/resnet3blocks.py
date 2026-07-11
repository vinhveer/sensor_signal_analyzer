import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_x = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding='same')
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding='same')
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn_z = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn_x(self.conv_x(x)))
        out = self.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))
        return self.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int=2, n_feature_maps: int=64, num_classes: int=4):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, n_feature_maps)
        self.block2 = ResidualBlock(n_feature_maps, n_feature_maps * 2)
        self.block3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(n_feature_maps * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.flatten(x)
        return self.out(x)