import torch
from torch import nn


class Embeddings(nn.Module):
    """
    Value and time features embedding module.

    Args:
        input_features: Number of input channels.
        output_features: Dimension of the output embedding.

    Attributes:
        conv_layer: 1D convolution layer for value embedding.
        linear_layer: Linear layer for the time features embedding.
        activation: Activation function.

    Returns:
        Embeddings.

    """

    def __init__(self, input_features: int, output_features: int):
        super(Embeddings, self).__init__()

        self.conv_layer = nn.Conv1d(in_channels=input_features, out_channels=output_features,
                                    kernel_size=3, padding=1, padding_mode='replicate')

        self.linear_layer = nn.Linear(in_features=3, out_features=output_features)

        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:

        x = self.conv_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_time = self.linear_layer(x_time)

        return x + x_time
