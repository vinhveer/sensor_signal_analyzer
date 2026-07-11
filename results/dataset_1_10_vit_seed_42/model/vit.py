import torch
import torch.nn as nn
def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0, b=2.0)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class FFN_SwiGLU(nn.Module):
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
        assert head_dim % 2 == 0, "head_dim must be even for RoPE1D"
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
    def __init__(self, seq_len: int = 1024, patch_size: int = 16, in_chans: int = 2, embed_dim: int = 64):
        super().__init__()
        assert seq_len % patch_size == 0, f"seq_len {seq_len} must be divisible by patch_size {patch_size}"
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).transpose(1, 2)


class Attention1D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
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

        num_patch_tokens = N - 1
        cos, sin = self.rope(num_patch_tokens, device=x.device, dtype=q.dtype)
        q = torch.cat((q[:, :, :1], apply_rope_1d(q[:, :, 1:], cos, sin)), dim=2)
        k = torch.cat((k[:, :, :1], apply_rope_1d(k[:, :, 1:], cos, sin)), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block1D(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, proj_drop: float = 0.0, attn_drop: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention1D(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = RMSNorm(dim)
        self.mlp = FFN_SwiGLU(d_model=dim, d_ff=int(dim * mlp_ratio * 2 / 3))
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer1D(nn.Module):
    def __init__(self, seq_len: int = 1024, patch_size: int = 16, in_chans: int = 2, num_classes: int = 4, embed_dim: int = 64, depth: int = 4, num_heads: int = 4, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_rate: float = 0.0, pos_drop_rate: float = 0.0, proj_drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_rate: float = 0.1):
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
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=1e-6)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
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
        x = self.head_drop(x[:, 0])
        return self.head(x)