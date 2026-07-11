import torch
import torch.nn as nn
import torch.nn.functional as F
class XCM1D(nn.Module):
    def __init__(self, in_channels: int=2, window: int=1024, num_classes: int =4, window_ratio: float=0.05, filters_num: int=64):
        super().__init__()
        kernel_size = max(1, int(window_ratio * window))
        self.conv2d = nn.Conv2d(1, filters_num, kernel_size=(kernel_size, 1), stride=(1, 1), padding="same")
        self.bn2d = nn.BatchNorm2d(filters_num)
        self.conv2d_reduced = nn.Conv2d(filters_num, 1, kernel_size=(1, 1), stride=(1, 1))

        self.conv1d = nn.Conv1d(in_channels, filters_num, kernel_size=kernel_size, stride=1, padding="same")
        self.bn1d = nn.BatchNorm1d(filters_num)
        self.conv1d_reduced = nn.Conv1d(filters_num, 1, kernel_size=1, stride=1)

        self.final_conv = nn.Conv1d(in_channels + 1, filters_num, kernel_size=kernel_size, stride=1, padding="same")
        self.final_bn = nn.BatchNorm1d(filters_num)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(filters_num, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.transpose(1, 2).unsqueeze(1)
        spatial = self.relu(self.bn2d(self.conv2d(spatial)))
        spatial = self.relu(self.conv2d_reduced(spatial)).squeeze(1)

        temporal = self.relu(self.bn1d(self.conv1d(x)))
        temporal = self.relu(self.conv1d_reduced(temporal)).transpose(1, 2)

        features = torch.cat([spatial, temporal], dim=2).transpose(1, 2)
        features = self.relu(self.final_bn(self.final_conv(features)))
        features = self.gap(features).squeeze(-1)
        return self.out(features)