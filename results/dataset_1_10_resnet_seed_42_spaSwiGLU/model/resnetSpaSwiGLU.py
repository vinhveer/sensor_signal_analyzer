import torch
import torch.nn as nn
import torch.nn.functional as F
class SpatialAttn1D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_attn = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)


class SwiGluFfn(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.01):
        super().__init__()
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        g, v = h[..., : self.d_ff], h[..., self.d_ff :]
        return self.dropout(self.linear2(g * torch.sigmoid(g) * v))


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
        self.spatial_attn = SpatialAttn1D()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn_x(self.conv_x(x)))
        out = self.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))
        out = self.spatial_attn(out)
        return self.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, in_channels: int=2, n_feature_maps: int=64, num_classes: int=4):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, n_feature_maps)
        self.block2 = ResidualBlock(n_feature_maps, n_feature_maps * 2)
        self.block3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2)
        d_model = n_feature_maps * 2
        self.ffn = SwiGluFfn(d_model=d_model, d_ff=int(d_model * 4 * 2 / 3))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(n_feature_maps * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.ffn(x.transpose(1, 2)).transpose(1, 2)
        x = self.gap(x)
        x = self.flatten(x)
        return self.out(x)