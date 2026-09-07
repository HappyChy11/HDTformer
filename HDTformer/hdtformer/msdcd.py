import torch
import torch.nn.functional as functional
from torch import nn


class DynamicConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_basis_kernels: int,
        basis_kernel_init: dict | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError("The paper implementation of DynamicConv1D is univariate")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_basis_kernels = num_basis_kernels
        # For z in R^[B,C_in], nn.Linear applies z W_1^T + b_1. Therefore,
        # W_1 has shape [K,C_in] and b_1 has shape [K]; broadcasting b_1 over
        # the batch produces K mixture logits for every input sample.
        self.weight_generator = nn.Linear(in_channels, num_basis_kernels)
        self.basis_kernels = nn.Parameter(
            torch.empty(
                num_basis_kernels,
                out_channels,
                in_channels,
                kernel_size,
            )
        )
        self.reset_basis_kernels(basis_kernel_init or {})

    def reset_basis_kernels(self, initialization: dict) -> None:
        distribution = initialization.get("distribution", "normal")
        if distribution == "normal":
            mean = float(initialization.get("mean", 0.0))
            std = float(initialization.get("std", 1.0))
            if std <= 0:
                raise ValueError("basis-kernel initialization std must be positive")
            nn.init.normal_(self.basis_kernels, mean=mean, std=std)
        elif distribution == "uniform":
            low = float(initialization.get("low", -1.0))
            high = float(initialization.get("high", 1.0))
            if low >= high:
                raise ValueError("basis-kernel initialization requires low < high")
            nn.init.uniform_(self.basis_kernels, a=low, b=high)
        else:
            raise ValueError(
                "basis-kernel distribution must be either 'normal' or 'uniform'"
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = inputs.shape
        mixture = self.weight_generator(inputs.mean(dim=-1)).softmax(dim=-1)
        kernels = torch.matmul(
            mixture, self.basis_kernels.view(self.num_basis_kernels, -1)
        ).view(batch_size * self.out_channels, self.in_channels, self.kernel_size)
        grouped_inputs = inputs.reshape(1, batch_size * channels, length)

        if self.kernel_size % 2 == 0:
            padding = (self.kernel_size // 2, self.kernel_size // 2 - 1)
        else:
            padding = (self.kernel_size // 2, self.kernel_size // 2)
        grouped_inputs = functional.pad(grouped_inputs, padding, value=0.0)
        outputs = functional.conv1d(grouped_inputs, kernels, groups=batch_size)
        return outputs.view(batch_size, self.out_channels, outputs.size(-1))


class MSDCD(nn.Module):
    """Multi-Scale Dynamic Convolutional Decomposition from the paper code."""

    def __init__(
        self,
        trend_kernel_size: int = 20,
        seasonal_kernel_size: int = 7,
        num_basis_kernels: int = 4,
        avg_pool_kernel_size: int = 3,
        basis_kernel_init: dict | None = None,
    ) -> None:
        super().__init__()
        self.trend_dynamic_conv = DynamicConv1D(
            1, 1, trend_kernel_size, num_basis_kernels, basis_kernel_init
        )
        self.seasonal_dynamic_conv = DynamicConv1D(
            1, 1, seasonal_kernel_size, num_basis_kernels, basis_kernel_init
        )
        self.avg_pool_kernel_size = avg_pool_kernel_size

    def average_pool(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.avg_pool_kernel_size % 2 == 0:
            padding = (
                self.avg_pool_kernel_size // 2,
                self.avg_pool_kernel_size // 2 - 1,
            )
        else:
            padding = (
                self.avg_pool_kernel_size // 2,
                self.avg_pool_kernel_size // 2,
            )
        padded = functional.pad(inputs, padding, mode="replicate")
        return functional.avg_pool1d(
            padded, kernel_size=self.avg_pool_kernel_size, stride=1
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        original = inputs.permute(0, 2, 1)
        initial_trend = self.average_pool(original).squeeze(1)
        detrended = original.squeeze(1) - initial_trend
        seasonal = self.seasonal_dynamic_conv(detrended.unsqueeze(1)).squeeze(1)
        deseasonalized = original.squeeze(1) - seasonal
        trend = self.trend_dynamic_conv(deseasonalized.unsqueeze(1)).squeeze(1)
        residual = original.squeeze(1) - trend - seasonal
        return trend, seasonal, residual, original.squeeze(1)
