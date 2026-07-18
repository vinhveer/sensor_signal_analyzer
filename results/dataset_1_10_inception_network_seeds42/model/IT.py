from torch import nn
import torch
import math
import torch.nn.functional as F
class Conv1dSame(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError(f"kernel_size phải > 0, nhận được {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride phải > 0, nhận được {stride}")

        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length_in = x.size(-1)
        length_out = math.ceil(length_in / self.stride)

        total_padding = max(
            0,
            (length_out - 1) * self.stride
            + self.kernel_size
            - length_in,
        )

        pad_left = total_padding // 2
        pad_right = total_padding - pad_left

        x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class InceptionModule(nn.Module):

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_size: int = 41,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
    ) -> None:
        super().__init__()

        if nb_filters <= 0:
            raise ValueError("nb_filters phải > 0")
        if bottleneck_size <= 0:
            raise ValueError("bottleneck_size phải > 0")
        if kernel_size <= 1:
            raise ValueError(
                "kernel_size phải > 1 vì code gốc sử dụng kernel_size - 1"
            )

        # Repo gốc nhận kernel_size=41 rồi dùng 41 - 1 = 40.
        base_kernel_size = int(kernel_size) - 1
        self.kernel_sizes = [
            base_kernel_size // (2**i)
            for i in range(3)
        ]

        if any(size <= 0 for size in self.kernel_sizes):
            raise ValueError(
                f"Các kernel sinh ra không hợp lệ: {self.kernel_sizes}"
            )

        should_use_bottleneck = use_bottleneck and in_channels > 1

        if should_use_bottleneck:
            self.bottleneck = Conv1dSame(
                in_channels=in_channels,
                out_channels=bottleneck_size,
                kernel_size=1,
                bias=False,
            )
            branch_in_channels = bottleneck_size
        else:
            self.bottleneck = nn.Identity()
            branch_in_channels = in_channels

        self.conv_branches = nn.ModuleList(
            [
                Conv1dSame(
                    in_channels=branch_in_channels,
                    out_channels=nb_filters,
                    kernel_size=size,
                    stride=1,
                    bias=False,
                )
                for size in self.kernel_sizes
            ]
        )

        # Nhánh MaxPool dùng input gốc, không dùng output bottleneck.
        self.max_pool = nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool_projection = Conv1dSame(
            in_channels=in_channels,
            out_channels=nb_filters,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        out_channels = 4 * nb_filters
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_output = self.bottleneck(x)

        branches = [
            branch(bottleneck_output)
            for branch in self.conv_branches
        ]

        pooled = self.max_pool(x)
        branches.append(self.pool_projection(pooled))

        output = torch.cat(branches, dim=1)
        output = self.batch_norm(output)
        return self.activation(output)


class ResidualShortcut(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.activation = nn.ReLU()

    def forward(
        self,
        residual_input: torch.Tensor,
        block_output: torch.Tensor,
    ) -> torch.Tensor:
        shortcut = self.projection(residual_input)
        return self.activation(shortcut + block_output)


class InceptionNetwork(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        nb_filters: int = 32,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        depth: int = 6,
        kernel_size: int = 41,
        bottleneck_size: int = 32,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels phải > 0")
        if num_classes <= 1:
            raise ValueError("num_classes phải > 1")
        if depth <= 0:
            raise ValueError("depth phải > 0")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.nb_filters = int(nb_filters)
        self.use_residual = bool(use_residual)
        self.use_bottleneck = bool(use_bottleneck)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.bottleneck_size = int(bottleneck_size)

        module_out_channels = 4 * self.nb_filters

        self.inception_modules = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        current_channels = self.in_channels
        residual_input_channels = self.in_channels

        for module_index in range(self.depth):
            self.inception_modules.append(
                InceptionModule(
                    in_channels=current_channels,
                    nb_filters=self.nb_filters,
                    kernel_size=self.kernel_size,
                    use_bottleneck=self.use_bottleneck,
                    bottleneck_size=self.bottleneck_size,
                )
            )

            current_channels = module_out_channels

            # Giống repo gốc: residual sau module thứ 3, 6, 9, ...
            if self.use_residual and module_index % 3 == 2:
                self.shortcuts.append(
                    ResidualShortcut(
                        in_channels=residual_input_channels,
                        out_channels=module_out_channels,
                    )
                )
                residual_input_channels = module_out_channels

        self.global_average_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(
            module_out_channels,
            self.num_classes,
        )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Input phải có shape "
                "[batch_size, channels, sequence_length]"
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Model yêu cầu {self.in_channels} channels, "
                f"nhưng input có {x.size(1)} channels"
            )

        residual_input = x
        shortcut_index = 0

        for module_index, inception_module in enumerate(
            self.inception_modules
        ):
            x = inception_module(x)

            if self.use_residual and module_index % 3 == 2:
                x = self.shortcuts[shortcut_index](
                    residual_input,
                    x,
                )
                residual_input = x
                shortcut_index += 1

        x = self.global_average_pooling(x)
        x = x.flatten(start_dim=1)

        return self.classifier(x)