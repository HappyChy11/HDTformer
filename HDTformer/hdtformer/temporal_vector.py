from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import nn


class TemporalVector(nn.Module):
    """Append element-wise linear and periodic transforms of ordered features.

    The input has shape ``[batch, sequence_length, input_feature_count]``.
    The paper configuration orders the five input features as RV, HEMA-1h,
    HEMA-12h, HEMA-1d, and RV-difference. It uses RV (index 0) in the linear
    branch and indices [0, 1, 2, 3] in the periodic branch.
    """

    def __init__(
        self,
        sequence_length: int,
        input_feature_count: int,
        linear_feature_indices: Sequence[int] = (0,),
        periodic_feature_indices: Sequence[int] = (0,),
    ) -> None:
        super().__init__()
        if sequence_length < 1 or input_feature_count < 1:
            raise ValueError("sequence_length and input_feature_count must be positive")
        linear_feature_indices = self._validate_indices(
            linear_feature_indices, input_feature_count, "linear_feature_indices"
        )
        periodic_feature_indices = self._validate_indices(
            periodic_feature_indices, input_feature_count, "periodic_feature_indices"
        )

        self.sequence_length = int(sequence_length)
        self.input_feature_count = int(input_feature_count)
        self.linear_feature_indices = linear_feature_indices
        self.periodic_feature_indices = periodic_feature_indices

        self.weights_linear = nn.Parameter(
            torch.empty(sequence_length, len(linear_feature_indices))
        )
        self.bias_linear = nn.Parameter(
            torch.empty(sequence_length, len(linear_feature_indices))
        )
        self.weights_periodic = nn.Parameter(
            torch.empty(sequence_length, len(periodic_feature_indices))
        )
        self.bias_periodic = nn.Parameter(
            torch.empty(sequence_length, len(periodic_feature_indices))
        )
        self.register_buffer(
            "_linear_indices",
            torch.tensor(linear_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_periodic_indices",
            torch.tensor(periodic_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.reset_parameters()

    @staticmethod
    def _validate_indices(
        indices: Sequence[int], input_feature_count: int, name: str
    ) -> tuple[int, ...]:
        if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
            raise TypeError(f"{name} must be an ordered sequence of integers")
        indices = tuple(indices)
        if any(isinstance(index, bool) or not isinstance(index, Integral) for index in indices):
            raise TypeError(f"{name} must contain only integer indices")
        if any(index < 0 or index >= input_feature_count for index in indices):
            raise ValueError(f"{name} contains an out-of-range index")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} must not contain duplicates")
        return tuple(int(index) for index in indices)

    @property
    def output_feature_count(self) -> int:
        return (
            self.input_feature_count
            + len(self.linear_feature_indices)
            + len(self.periodic_feature_indices)
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weights_linear, mean=0.0, std=1.0)
        nn.init.normal_(self.bias_linear, mean=0.0, std=1.0)
        nn.init.normal_(self.weights_periodic, mean=0.0, std=1.0)
        nn.init.normal_(self.bias_periodic, mean=0.0, std=1.0)

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

        linear_input = inputs.index_select(2, self._linear_indices)
        linear = self.weights_linear * linear_input + self.bias_linear

        periodic_inputs = inputs.index_select(2, self._periodic_indices)
        periodic = torch.sin(
            self.weights_periodic.unsqueeze(0) * periodic_inputs
            + self.bias_periodic.unsqueeze(0)
        )
        return torch.cat([inputs, linear, periodic], dim=-1)
