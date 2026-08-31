import torch
from torch import nn


class TemporalVector(nn.Module):
    """Learn linear and periodic representations from the short-term input."""

    def __init__(self, sequence_length: int, periodic_feature_count: int) -> None:
        super().__init__()
        self.weights_linear = nn.Parameter(torch.randn(sequence_length))
        self.bias_linear = nn.Parameter(torch.randn(sequence_length))
        self.weights_periodic = nn.Parameter(
            torch.randn(sequence_length, periodic_feature_count)
        )
        self.bias_periodic = nn.Parameter(
            torch.randn(sequence_length, periodic_feature_count)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first_feature = inputs[:, :, 0]
        linear = self.weights_linear * first_feature + self.bias_linear
        linear = linear.unsqueeze(-1)
        feature_count = self.weights_periodic.shape[1]
        periodic = torch.sin(
            self.weights_periodic * inputs[:, :, :feature_count]
            + self.bias_periodic
        )
        return torch.cat([inputs, linear, periodic], dim=-1)
