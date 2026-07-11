import torch
import torch.nn as nn
def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0, b=2.0)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return x.div(keep_prob) * x.new_empty(shape).bernoulli_(keep_prob)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ── ResNet1D branch ───────────────────────────────────────────────────────────

class SE1D(nn.Module):
    def __init__(self, channels: int, rd_ratio: float = 1.0 / 16):
        super().__init__()
        squeeze_channels = max(1, int(channels * rd_ratio))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, squeeze_channels, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(squeeze_channels, channels, kernel_size=1, bias=False)
        self.scale_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_x = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding='same')
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding='same')
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same')
        self.bn_z = nn.BatchNorm1d(out_channels)
        self.se = SE1D(out_channels)
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
        out = self.se(out)
        return self.relu(out + self.shortcut(x))


class ResNet1DExtractor(nn.Module):
    """ResNet1D without classification head — returns GAP feature vector."""
    def __init__(self, in_channels: int=2, n_feature_maps: int=64):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, n_feature_maps)
        self.block2 = ResidualBlock(n_feature_maps, n_feature_maps * 2)
        self.block3 = ResidualBlock(n_feature_maps * 2, n_feature_maps * 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out_dim = n_feature_maps * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.flatten(self.gap(x))


# ── ViT1D branch ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class FFNSwiGLU(nn.Module):
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


class RoPE1D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._cache: dict = {}

    def forward(self, seq_len: int, device, dtype):
        key = (seq_len, device, dtype)
        if key not in self._cache:
            positions = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(positions, self.inv_freq.to(device=device, dtype=dtype))
            self._cache[key] = (freqs.cos(), freqs.sin())
        return self._cache[key]


def apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class PatchEmbed1D(nn.Module):
    def __init__(self, seq_len: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert seq_len % patch_size == 0
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).transpose(1, 2)


class Attention1D(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.rope = RoPE1D(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos, sin = self.rope(N - 1, device=x.device, dtype=q.dtype)
        q = torch.cat((q[:, :, :1], apply_rope_1d(q[:, :, 1:], cos, sin)), dim=2)
        k = torch.cat((k[:, :, :1], apply_rope_1d(k[:, :, 1:], cos, sin)), dim=2)
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


class Block1D(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, proj_drop: float = 0.0, attn_drop: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention1D(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = RMSNorm(dim)
        self.mlp = FFNSwiGLU(d_model=dim, d_ff=int(dim * mlp_ratio * 2 / 3))
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class ViT1DExtractor(nn.Module):
    """ViT1D without classification head — returns CLS token feature vector."""
    def __init__(self, seq_len: int, patch_size: int, in_chans: int, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, pos_drop_rate: float = 0.0, proj_drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbed1D(seq_len=seq_len, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(pos_drop_rate)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(*[
            Block1D(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_drop=proj_drop_rate, attn_drop=attn_drop_rate, drop_path_rate=dpr[i])
            for i in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)
        self.out_dim = embed_dim
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=1e-6)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]


# ── Hybrid model ──────────────────────────────────────────────────────────────

class HybridCNNTransformer(nn.Module):
  
    def __init__(self, in_chans: int=2, seq_len: int=1024, num_classes: int=4, n_feature_maps: int = 64, patch_size: int = 16, embed_dim: int = 64, depth: int = 4, num_heads: int = 4, proj_dim: int = 128, dropout: float = 0.2, drop_path_rate: float = 0.1):
        super().__init__()
        self.resnet = ResNet1DExtractor(in_channels=in_chans, n_feature_maps=n_feature_maps)
        self.vit = ViT1DExtractor(seq_len=seq_len, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads, drop_path_rate=drop_path_rate)

        self.res_proj = nn.Sequential(
            nn.Linear(self.resnet.out_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(self.vit.out_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_feat = self.res_proj(self.resnet(x))
        vit_feat = self.vit_proj(self.vit(x))
        fused = torch.cat([res_feat, vit_feat], dim=1)
        return self.classifier(fused)