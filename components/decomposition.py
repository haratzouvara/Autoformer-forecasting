import torch
from torch import nn
from typing import Tuple


class DecomposeSequence(nn.Module):
    """
    Decompose time series data into trend and season components using moving average.

    Args:
        kernel_size: Size of the moving average kernel.
        stride: Stride of the moving average operation.

    Attributes:
        kernel_size: Size of the moving average kernel.
        stride: Stride of the moving average operation.
        moving_average: Moving average layer.
        padding_left: Left padding size.
        padding_right: Right padding size.

    """

    def __init__(self, kernel_size, stride):
        super(DecomposeSequence, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.moving_average = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self.padding_left = (self.kernel_size - 1) // 2
        if self.kernel_size % 2 == 0:
            self.padding_right = self.kernel_size // 2
        else:
            self.padding_right = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DecomposeSequence module.

        Args:
            x: Input time series data.

        Returns:
            Tuple containing season and trend components.

        """

        front = x[:, 0:1, :].repeat(1, self.padding_left, 1)
        end = x[:, -1:, :].repeat(1, self.padding_right, 1)

        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.moving_average(x_pad.permute(0, 2, 1))
        trend = trend.permute(0, 2, 1)
        season = x - trend
        return season, trend
