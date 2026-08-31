from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class TemporalVector(nn.Module):
    """Extract one linear feature and explicitly selected periodic features.

    The input has shape ``[batch, sequence_length, input_feature_count]``.
    The paper configuration orders the five input features as RV, HEMA-1h,
    HEMA-12h, HEMA-1d, and RV-difference. It uses RV (index 0) in the linear
    branch and indices [0, 1, 2, 3] in the periodic branch.
    """

    def __init__(
        self,
        sequence_length: int,
        input_feature_count: int,
        linear_feature_index: int = 0,
        periodic_feature_indices: Sequence[int] = (0,),
        init_mean: float = 0.0,
        init_std: float = 1.0,
    ) -> None:
        super().__init__()
        if sequence_length < 1 or input_feature_count < 1:
            raise ValueError("sequence_length and input_feature_count must be positive")
        if not 0 <= linear_feature_index < input_feature_count:
            raise ValueError("linear_feature_index is outside the input feature range")

        periodic_feature_indices = tuple(int(index) for index in periodic_feature_indices)
        if len(set(periodic_feature_indices)) != len(periodic_feature_indices):
            raise ValueError("periodic_feature_indices must not contain duplicates")
        if any(index < 0 or index >= input_feature_count for index in periodic_feature_indices):
            raise ValueError("periodic_feature_indices contains an out-of-range index")
        if init_std <= 0:
            raise ValueError("init_std must be positive")

        self.sequence_length = int(sequence_length)
        self.input_feature_count = int(input_feature_count)
        self.linear_feature_index = int(linear_feature_index)
        self.periodic_feature_indices = periodic_feature_indices

        # These parameter shapes match the experiment implementation and keep
        # the released checkpoints loadable.
        self.weights_linear = nn.Parameter(torch.empty(sequence_length))
        self.bias_linear = nn.Parameter(torch.empty(sequence_length))
        self.weights_periodic = nn.Parameter(
            torch.empty(sequence_length, len(periodic_feature_indices))
        )
        self.bias_periodic = nn.Parameter(
            torch.empty(sequence_length, len(periodic_feature_indices))
        )
        self.register_buffer(
            "_periodic_indices",
            torch.tensor(periodic_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.reset_parameters(init_mean=init_mean, init_std=init_std)

    @property
    def output_feature_count(self) -> int:
        return self.input_feature_count + 1 + len(self.periodic_feature_indices)

    def reset_parameters(self, init_mean: float = 0.0, init_std: float = 1.0) -> None:
        nn.init.normal_(self.weights_linear, mean=init_mean, std=init_std)
        nn.init.normal_(self.bias_linear, mean=init_mean, std=init_std)
        nn.init.normal_(self.weights_periodic, mean=init_mean, std=init_std)
        nn.init.normal_(self.bias_periodic, mean=init_mean, std=init_std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("TemporalVector inputs must have shape [batch, length, features]")
        if inputs.size(1) != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {inputs.size(1)}"
            )
        if inputs.size(2) != self.input_feature_count:
            raise ValueError(
                f"Expected {self.input_feature_count} input features, got {inputs.size(2)}"
            )

        linear_input = inputs[:, :, self.linear_feature_index]
        linear = self.weights_linear * linear_input + self.bias_linear
        linear = linear.unsqueeze(-1)

        periodic_inputs = inputs.index_select(2, self._periodic_indices)
        periodic = torch.sin(
            self.weights_periodic.unsqueeze(0) * periodic_inputs
            + self.bias_periodic.unsqueeze(0)
        )
        return torch.cat([inputs, linear, periodic], dim=-1)
