import torch
import torch.nn as nn
import torch.nn.functional as F
def _build_hybrid_filters(kernel_sizes: list) -> list:
    """Build 1D fixed filter tensors for the hybrid layer (3 types)."""
    filters = []
    for k in kernel_sizes:
        f = torch.ones(k)
        f[torch.arange(k) % 2 == 0] = -1.0
        filters.append(f)
    for k in kernel_sizes:
        f = torch.ones(k)
        f[torch.arange(k) % 2 > 0] = -1.0
        filters.append(f)
    for k in kernel_sizes[1:]:
        kl = k + k // 2
        f = torch.zeros(kl)
        n = k // 4
        if n > 0:
            t = torch.linspace(0, 1, n + 1)[1:]
            fl, fr = t ** 2, (t ** 2).flip(0)
            f[0:n] = -fl
            f[n:k // 2] = -fr
            f[k // 2:3 * k // 4] = 2 * fl
            f[3 * k // 4:k] = 2 * fr
            f[k:k + n] = -fl
            f[k + n:] = -fr
        filters.append(f)
    return filters


class _FixedDepthwiseConv1D(nn.Module):
    """Non-trainable depthwise Conv1d with a fixed filter applied to all channels."""
    def __init__(self, in_ch: int, filter_1d: torch.Tensor):
        super().__init__()
        k = filter_1d.shape[0]
        weight = filter_1d.float().view(1, 1, k).expand(in_ch, 1, k).clone()
        self.register_buffer("weight", weight)
        self.pad_left = (k - 1) // 2
        self.pad_right = (k - 1) - self.pad_left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad_left, self.pad_right))
        return F.conv1d(x, self.weight, groups=self.weight.shape[0])


class HybridLayer1D(nn.Module):
    """Fixed non-trainable depthwise filters (alternating ±1 and quadratic peak shapes)."""
    _KERNEL_SIZES = [2, 4, 8, 16, 32, 64]

    def __init__(self, in_ch: int):
        super().__init__()
        self.convs = nn.ModuleList(
            [_FixedDepthwiseConv1D(in_ch, f) for f in _build_hybrid_filters(self._KERNEL_SIZES)]
        )
        self.act = nn.ReLU(inplace=True)
        self.out_ch = in_ch * len(self.convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(torch.cat([c(x) for c in self.convs], dim=1))


class SeparableConv1D(nn.Module):
    """Depthwise + pointwise Conv1d (mirrors Keras SeparableConv1D)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size,
                            groups=in_ch, padding='same', dilation=dilation, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class InceptionModule1D(nn.Module):
    """Multi-scale SeparableConv1D branches + optional HybridLayer → BN → ReLU."""
    def __init__(self, in_ch: int, n_filters: int, kernel_size: int,
                 use_hybrid: bool = True, use_multiplexing: bool = True):
        super().__init__()
        n_convs = 3 if use_multiplexing else 1
        n_f = n_filters if use_multiplexing else n_filters * 3
        self.sep_convs = nn.ModuleList(
            [SeparableConv1D(in_ch, n_f, max(1, kernel_size // (2 ** i))) for i in range(n_convs)]
        )
        self.hybrid = HybridLayer1D(in_ch) if use_hybrid else None
        hyb_ch = self.hybrid.out_ch if use_hybrid else 0
        self.out_ch = n_f * n_convs + hyb_ch
        self.bn = nn.BatchNorm1d(self.out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [c(x) for c in self.sep_convs]
        if self.hybrid is not None:
            parts.append(self.hybrid(x))
        return self.act(self.bn(torch.cat(parts, dim=1)))


class FCNModule1D(nn.Module):
    """SeparableConv1D + BN + ReLU."""
    def __init__(self, in_ch: int, n_filters: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = SeparableConv1D(in_ch, n_filters, max(1, kernel_size), dilation=dilation)
        self.bn = nn.BatchNorm1d(n_filters)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class LiteEMV1D(nn.Module):
    """
    PyTorch 1D port of LITEMV for input [B, in_channels, seq_len].

    Mirrors build_model():
      InceptionModule1D (multi-scale SeparableConv + HybridLayer)
      → FCNModule1D × 2 (with increasing dilation)
      → GlobalAvgPool → Linear(num_classes)
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        n_filters: int = 64,
        kernel_size: int = 61,
        use_custom_filters: bool = True,
        use_dilation: bool = True,
        use_multiplexing: bool = True,
    ):
        super().__init__()
        k = kernel_size - 1
        self.inception = InceptionModule1D(
            in_ch=in_channels, n_filters=n_filters, kernel_size=k,
            use_hybrid=use_custom_filters, use_multiplexing=use_multiplexing,
        )
        k //= 2
        fcn_in = self.inception.out_ch
        self.fcn_modules = nn.ModuleList()
        for i in range(2):
            dilation = 2 ** (i + 1) if use_dilation else 1
            self.fcn_modules.append(
                FCNModule1D(fcn_in, n_filters, kernel_size=max(1, k // (2 ** i)), dilation=dilation)
            )
            fcn_in = n_filters
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(n_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inception(x)
        for fcn in self.fcn_modules:
            x = fcn(x)
        return self.out(self.flatten(self.gap(x)))