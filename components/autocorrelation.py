import math
import torch
from torch import nn


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation module based on the paper https://arxiv.org/pdf/2106.13008.pdf
    and the corresponding pseudocode (Algorithms 3, 4).

    This module calculates autocorrelation using FFT
    and incorporates a speed-up version for training and inference phases.
    The time delays aggregation is performed based on the methodology described in the corresponding paper.

    Args:
        factor: A scaling factor for determining the number of top correlations to consider.
        flag: 'train' for training phase and 'test' for inference phase.

    Methods:
        time_delay_agg_train(values, correlation):
            Performs time delay aggregation during the training phase.

        time_delay_agg_inference(values, correlation):
            Performs time delay aggregation during the inference phase.

    Attributes:
        factor: Scaling factor for determining the number of top correlations to consider.
        flag: 'train' for training phase and 'test' for inference phase.
    """

    def __init__(self, heads: int, factor: int, flag: str):
        super(AutoCorrelation, self).__init__()
        self.heads = heads
        self.factor = factor
        self.flag = flag

    def time_delay_agg_training(self, value: torch.Tensor, correlation: torch.Tensor) -> torch.Tensor:
        """
        Time delay aggregation during the training phase.

        Args:
            value: Input tensor.
            correlation: Autocorrelation tensor.

        Returns:
            Aggregated tensor based on time delays.
        """

        b, l, h, c = value.shape

        top_k = int(self.factor * math.log(l))

        mean_value = torch.mean(torch.mean(correlation, dim=-1), dim=-1)
        index_topk = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights_topk = torch.stack([mean_value[:, index_topk[i]] for i in range(top_k)], dim=-1)

        weighted_correlations = torch.softmax(weights_topk, dim=-1)

        delay_aggregation = torch.zeros_like(value)
        for i in range(top_k):
            roll_sequence = torch.roll(value, -int(index_topk[i]), dims=1)
            delay_aggregation = (delay_aggregation + roll_sequence *
                                 weighted_correlations[:, i].view(b, 1, 1, 1).expand(b, l, h, c))
        return delay_aggregation

    def time_delay_agg_inference(self, value: torch.Tensor, correlation: torch.Tensor) -> torch.Tensor:
        """
        Time delay aggregation during the inference phase.

        Args:
            value: Input tensor.
            correlation: Autocorrelation tensor.

        Returns:
            Aggregated tensor based on time delays.
        """

        b, l, h, c = value.shape

        top_k = int(self.factor * math.log(l))

        mean_value = torch.mean(torch.mean(correlation, dim=-1), dim=-1)
        index_topk = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights_topk = torch.stack([mean_value[:, index_topk[i]] for i in range(top_k)], dim=-1)

        weighted_correlations = torch.softmax(weights_topk, dim=-1)

        repeat_value = value.repeat(1, 2, 1, 1)
        Index = torch.arange(l).view(1, l, 1, 1).expand(b, l, h, c)

        delay_aggregation = torch.zeros_like(value)
        for i in range(top_k):

            delay = Index + index_topk[i]
            gather_sequence = torch.gather(repeat_value, index=delay, dim=1)
            delay_aggregation = (delay_aggregation + gather_sequence *
                                 weighted_correlations[:, i].view(b, 1, 1, 1).expand(b, l, h, c))

        return delay_aggregation

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AutoCorrelation module.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.

        Returns:
            New value tensor after applying AutoCorrelation.
        """

        q_b, q_l, q_h, q_c = query.shape
        v_b, v_l, v_h, v_c = value.shape

        #  we need it for the decoder case,
        #  where the queries come from the encoder and the keys and values from the decoder input
        if q_l > v_l:
            zeros = torch.zeros_like(query[:, :(q_l - v_l), :])
            value = torch.cat([value, zeros], dim=1)
            key = torch.cat([key, zeros], dim=1)
        else:
            value = value[:, :q_l, :, :]
            key = key[:, :q_l, :, :]

        query_fft = torch.fft.rfft(query.permute(0, 2, 3, 1), dim=-1)
        key_fft = torch.fft.rfft(key.permute(0, 2, 3, 1), dim=-1)

        correlation = torch.fft.irfft(query_fft * torch.conj(key_fft), n=q_l, dim=-1)
        correlation = correlation.permute(0, 3, 1, 2)

        if self.flag == 'train':
            autocorrelation_value = self.time_delay_agg_training(value, correlation)
        elif self.flag == 'test':
            autocorrelation_value = self.time_delay_agg_inference(value, correlation)
        else:
            raise ValueError("The flag should be set to either 'train' or 'test'.")

        return autocorrelation_value


class AutoCorrelationLayer(nn.Module):
    """
    AutoCorrelationLayer module combines linear transformations with AutoCorrelation.

    This layer takes the query, key, and value, as input and applies linear transformation and
    computes autocorrelation using the AutoCorrelation module. The final output is obtained
    by applying another linear transformation.

    Args:
        input_features: Number of input features.
        hidden_features: Number of hidden features in linear transformations.
        factor: A scaling factor for determining the number of top correlations to consider.
        flag: 'train' for training phase and 'test' for inference phase.

    Attributes:
        correlation: AutoCorrelation module.
        linear_query: Linear transformation for query.
        linear_key: Linear transformation for key.
        linear_value: Linear transformation for value.
        linear_output: Linear transformation for the final output.
    """

    def __init__(self, input_features: int, hidden_features: int, heads: int, factor: int, flag: str):
        super(AutoCorrelationLayer, self).__init__()

        # we need it for the case where hidden_features
        # is not perfectly divisible by heads
        heads_features = (hidden_features // heads) * heads

        self.heads = heads
        self.correlation = AutoCorrelation(heads=heads, factor=factor, flag=flag)
        self.linear_query = nn.Linear(input_features, heads_features)
        self.linear_key = nn.Linear(input_features, heads_features)
        self.linear_value = nn.Linear(input_features, heads_features)
        self.linear_output = nn.Linear(heads_features, hidden_features)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AutoCorrelationProcessor module.

        Args:
            query: Input tensor for query.
            key: Input tensor for key.
            value: Input tensor for value.

        Returns:
            Output tensor after applying AutoCorrelationProcessor.
        """

        q_b, q_l, _ = query.shape
        k_b, k_l, _ = key.shape
        v_b, v_l, _ = value.shape

        query = self.linear_query(query).view(q_b, q_l, self.heads, -1)
        key = self.linear_key(key).view(k_b, k_l, self.heads, -1)
        value = self.linear_value(value).view(v_b, v_l, self.heads, -1)

        correlation_output = self.correlation(query, key, value)
        correlation_output = correlation_output.view(q_b, q_l, -1)

        return self.linear_output(correlation_output)
