from typing import Tuple
import torch
from torch import nn
from .decomposition import DecomposeSequence
from .autocorrelation import AutoCorrelationLayer
from .embeddings import Embeddings


class Encoder(nn.Module):
    """
        EncoderLayer module with AutoCorrelation and Decomposition
        based on the paper https://arxiv.org/pdf/2106.13008.pdf.

        This layer consists of AutoCorrelation, Decomposition, 1D Convolution, and LeakyReLU activation.
        It is used as a building block in the encoder of the AutoFormer model.

        Args:
            input_features: Number of input features.
            hidden_features: Number of hidden features in the correlation layer.
            decomposition_kernel: Size of the decomposition kernel.
            factor: A scaling factor for AutoCorrelation.
            flag: 'train' for training phase and 'test' for inference phase.

        Attributes:
            ac_layer: AutoCorrelationLayer module.
            decomp_1: DecomposeSequence module for the first decomposition.
            decomp_2: DecomposeSequence module for the second decomposition.
            conv_layer_1: 1D Convolution layer for the FF layer.
            conv_layer_2: 1D Convolution layer for the FF layer.
            activation: Activation function.
            layernorm: Normalize season output.
        """

    def __init__(self, input_features: int, hidden_features: int, convolution_features: int, decomposition_kernel: int,
                 heads: int, factor: int, flag: str):
        super(Encoder, self).__init__()

        self.ac_layer = AutoCorrelationLayer(input_features=input_features, hidden_features=hidden_features,
                                             heads=heads, factor=factor, flag=flag)

        self.decomp_1 = DecomposeSequence(kernel_size=decomposition_kernel, stride=1)
        self.decomp_2 = DecomposeSequence(kernel_size=decomposition_kernel, stride=1)

        self.conv_layer_1 = nn.Conv1d(in_channels=hidden_features, out_channels=convolution_features,
                                      kernel_size=3, padding=1, padding_mode='replicate', bias=False)
        self.conv_layer_2 = nn.Conv1d(in_channels=convolution_features, out_channels=hidden_features,
                                      kernel_size=3, padding=1, padding_mode='replicate', bias=False)

        self.activation = nn.LeakyReLU()
        self.layernorm = nn.LayerNorm(hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EncoderLayer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the EncoderLayer.
        """

        autocorr_x = self.ac_layer(x, x, x) + x
        season_first, _ = self.decomp_1(autocorr_x)
        conv_output = self.conv_layer_1(season_first.permute(0, 2, 1))
        conv_output = self.activation(conv_output)
        conv_output = self.conv_layer_2(conv_output)
        conv_output = self.activation(conv_output)

        conv_output = conv_output.permute(0, 2, 1) + season_first
        encoder_layer_output, _ = self.decomp_2(conv_output)
        encoder_layer_output = self.layernorm(encoder_layer_output)

        return encoder_layer_output


class Decoder(nn.Module):
    """
     DecoderLayer module with Decomposition, AutoCorrelation, Convolution, and Linear layers.

     This layer represents a component of the decoder architecture. It combines AutoCorrelation,
     Decomposition, Convolution, and Linear transformations to generate the final decoder output.

     Args:
         input_features: Number of input features.
         hidden_features: Number of hidden features in the layers.
         convolution_features: Number of the hidden convolutional features in the FF layer.
         decomposition_kernel: Size of the decomposition kernel.
         factor: A scaling factor for AutoCorrelation.
         flag: 'train' for training phase and 'test' for inference phase.

     Attributes:
         decomp_init: Initial DecomposeSequence module.
         decomp_1: DecomposeSequence module for the first decomposition.
         decomp_2: DecomposeSequence module for the second decomposition.
         decomp_3: DecomposeSequence module for the third decomposition.
         ac_layer_1: AutoCorrelationLayer module for the first autocorrelation.
         ac_layer_2: AutoCorrelationLayer module for the second autocorrelation.
         linear_layer_1: Linear layer for the first trend linear transformation.
         linear_layer_2: Linear layer for the second trend linear transformation.
         linear_layer_3: Linear layer for the third trend linear transformation.
         conv_layer_1: 1D Convolution layer for the first convolutional transformation in the FF layer.
         conv_layer_2: 1D Convolution layer for the second convolutional transformation in the FF layer.
         linear_layer_season: 1D Convolution layer for the final seasonal output.
         activation: Activation function.
         layernorm: Normalize season output.
     """

    def __init__(self, input_features: int, hidden_features: int, convolution_features: int,
                 decomposition_kernel: int, heads: int, factor: int, flag: str):
        super(Decoder, self).__init__()

        self.decomp_init = DecomposeSequence(kernel_size=decomposition_kernel, stride=1)
        self.decomp_1 = DecomposeSequence(kernel_size=decomposition_kernel, stride=1)
        self.decomp_2 = DecomposeSequence(kernel_size=decomposition_kernel, stride=1)
        self.decomp_3 = DecomposeSequence(kernel_size=decomposition_kernel, stride=1)

        self.ac_layer_1 = AutoCorrelationLayer(input_features=input_features, hidden_features=hidden_features,
                                               heads=heads, factor=factor, flag=flag)
        self.ac_layer_2 = AutoCorrelationLayer(input_features=hidden_features, hidden_features=hidden_features,
                                               heads=heads, factor=factor, flag=flag)

        self.linear_layer_1 = nn.Linear(hidden_features, hidden_features, bias=False)
        self.linear_layer_2 = nn.Linear(hidden_features, hidden_features, bias=False)
        self.linear_layer_3 = nn.Linear(hidden_features, hidden_features, bias=False)

        self.conv_layer_1 = nn.Conv1d(in_channels=hidden_features, out_channels=convolution_features,
                                      kernel_size=3, padding=1, padding_mode='replicate', bias=False)
        self.conv_layer_2 = nn.Conv1d(in_channels=convolution_features, out_channels=hidden_features,
                                      kernel_size=3, padding=1, padding_mode='replicate', bias=False)

        self.linear_layer_season = nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=False)

        self.activation = nn.LeakyReLU()

        self.layernorm = nn.LayerNorm(hidden_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seasonal_init: torch.Tensor, trend_init: torch.Tensor, encoder_output: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
         Forward pass of the DecoderLayer.

         Args:
             seasonal_init: Initial seasonal component.
             trend_init: Initial trend component.
             encoder_output: Output from the encoder.

         Returns:
             Final output of the decoder layer.
         """

        autocorr_1 = self.ac_layer_1(seasonal_init, seasonal_init, seasonal_init) + seasonal_init
        season_1, trend_1 = self.decomp_1(autocorr_1)
        trend_1 = self.linear_layer_1(trend_1)

        autocorr_2 = self.ac_layer_2(season_1, encoder_output, encoder_output) + season_1
        season_2, trend_2 = self.decomp_2(autocorr_2)
        trend_2 = self.linear_layer_2(trend_2)

        conv_output = self.conv_layer_1(season_2.permute(0, 2, 1))
        conv_output = self.activation(conv_output)
        conv_output = self.activation(self.conv_layer_2(conv_output).permute(0, 2, 1)) + season_2

        season_3, trend_3 = self.decomp_3(conv_output)
        season_output = self.layernorm(season_3)
        season_output = self.linear_layer_season(season_output)

        trend_3 = self.linear_layer_3(trend_3)

        trend_output = trend_init + trend_1 + trend_2 + trend_3

        return season_output, trend_output


class AutoFormer(nn.Module):
    """
    AutoFormer module combining Encoder and Decoder layers for time series forecasting.

    This module integrates an EncoderLayer and a DecoderLayer to perform time series forecasting
    using the AutoFormer architecture.

    Args:
        input_features: Number of input features in the time series.
        hidden_features: Number of hidden features in the layers.
        convolution_features: Number of convolutional features in the decoder FF layer.
        output_features: Number of output features.
        decomposition_kernel: Size of the decomposition kernel.
        mask_len: Length of the masking sequence in the DecoderLayer.
        output_len: Length of the output sequence in the DecoderLayer.
        factor: A scaling factor for AutoCorrelation in the layers.
        flag: 'train' for training phase and 'test' for inference phase.

    Attributes:
        output_len: Length of the output sequence.
        mask_len: Length of the masking sequence.
        decomp_init: Initial DecomposeSequence module.
        emb: Embedding module.
        encoder: EncoderLayer module.
        decoder: DecoderLayer module.
        final_linear_layer: Linear layer for the final prediction.
    """

    def __init__(self, input_features: int, hidden_features: int, convolution_features: int,
                 output_features: int, decomposition_kernel: int, mask_len: int, output_len: int, factor: int,
                 heads: int, flag: str):
        super(AutoFormer, self).__init__()

        self.output_len = output_len
        self.mask_len = mask_len
        self.output_features = output_features

        self.decomp_init = DecomposeSequence(kernel_size=decomposition_kernel, stride=1)
        self.emb = Embeddings(input_features=input_features, output_features=hidden_features)

        self.encoder = Encoder(input_features=hidden_features, hidden_features=hidden_features,
                               convolution_features=convolution_features,
                               decomposition_kernel=decomposition_kernel,
                               heads=heads, factor=factor, flag=flag)
        self.decoder = Decoder(input_features=hidden_features, hidden_features=hidden_features,
                               convolution_features=convolution_features,
                               decomposition_kernel=decomposition_kernel,
                               heads=heads, factor=factor, flag=flag)

        self.final_linear_layer = nn.Linear(in_features=hidden_features, out_features=output_features, bias=False)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, x_time: torch.Tensor, y_time: torch.Tensor) -> torch.Tensor:
        """
         Forward pass of the AutoFormer.

         Args:
             x: Input time series data.
             x_time: Input time features.
             y_time: Input time features for the output.

         Returns:
             Final prediction for the time series.
         """

        season, trend = self.decomp_init(x)
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.output_len, 1)
        zeros = torch.zeros([x.shape[0], self.output_len, self.output_features])

        seasonal_init = torch.cat([season[:, -self.mask_len:, :], zeros], dim=1)  # (mask + output) length
        trend_init = torch.cat([trend[:, -self.mask_len:, :], mean], dim=1)

        x_emb = self.emb(x, x_time)
        seasonal_init_emb = self.emb(seasonal_init, y_time)

        encoder_output = self.encoder(x_emb)
        decoder_season, decoder_trend = self.decoder(seasonal_init_emb, trend_init, encoder_output)

        final_prediction = decoder_season + decoder_trend
        final_prediction = self.activation(final_prediction)
        final_prediction = self.final_linear_layer(final_prediction)

        final_prediction = final_prediction[:, -self.output_len:, :]

        return final_prediction
