from __future__ import annotations

import torch
from torch import nn

from .msdcd import MSDCD
from .temporal_vector import TemporalVector
from .transformer import (
    PositionalEncoding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)


class ShortTermDependency(nn.Module):
    def __init__(
        self,
        short_lookback: int,
        embed_dim: int,
        dense_dim: int,
        num_heads: int,
        dropout: float,
        num_blocks: int,
        positional_encoding: bool,
        use_mask: bool,
        max_len: int,
        temporal_vector_config: dict,
    ) -> None:
        super().__init__()
        self.use_positional_encoding = positional_encoding
        self.use_mask = use_mask
        self.temporal_vector = TemporalVector(
            sequence_length=short_lookback,
            **temporal_vector_config,
        )
        self.embedding = nn.Linear(
            self.temporal_vector.output_feature_count, embed_dim
        )
        self.position = PositionalEncoding(embed_dim, max_len)
        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, dense_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.decoders = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, dense_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        # PyTorch ignores LSTM dropout for a single recurrent layer.
        self.lstm = nn.LSTM(embed_dim, 4 * embed_dim, batch_first=True)

    @staticmethod
    def generate_mask(size: int, output_length: int, device: torch.device):
        mask = torch.zeros(size, size, device=device)
        mask[:, -output_length:] = float("-inf")
        for offset in range(1, output_length + 1):
            mask[-offset, -offset:] = 0
        return mask

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mask = None
        if self.use_mask:
            mask = self.generate_mask(inputs.size(1), 1, inputs.device)

        outputs = self.embedding(self.temporal_vector(inputs))
        if self.use_positional_encoding:
            outputs = self.position(outputs)
        for encoder in self.encoders:
            outputs = encoder(outputs)

        encoder_outputs = outputs
        decoder_outputs = encoder_outputs
        if self.use_positional_encoding:
            decoder_outputs = self.position(decoder_outputs)
        for decoder in self.decoders:
            decoder_outputs = decoder(decoder_outputs, encoder_outputs, mask)

        outputs, _ = self.lstm(decoder_outputs)
        return outputs[:, -1:, :]


class LongTermView(nn.Module):
    def __init__(self, embed_dim: int, msdcd_config: dict) -> None:
        super().__init__()
        self.msdcd = MSDCD(**msdcd_config)
        self.embedding = nn.Linear(3, embed_dim // 2)
        self.lstm = nn.LSTM(embed_dim // 2, 2 * embed_dim, batch_first=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        trend, seasonal, _residual, original = self.msdcd(inputs)
        features = torch.stack([trend, seasonal, original], dim=-1)
        outputs, _ = self.lstm(self.embedding(features))
        return outputs[:, -1:, :]


class HDTformer(nn.Module):
    def __init__(
        self,
        short_lookback: int,
        embed_dim: int = 32,
        dense_dim: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_blocks: int = 4,
        positional_encoding: bool = True,
        use_mask: bool = False,
        max_len: int = 5000,
        temporal_vector: dict | None = None,
        msdcd: dict | None = None,
    ) -> None:
        super().__init__()
        temporal_vector = temporal_vector or {
            "input_feature_count": 1,
            "linear_feature_index": 0,
            "periodic_feature_indices": [0],
        }
        msdcd = msdcd or {}
        self.std = ShortTermDependency(
            short_lookback,
            embed_dim,
            dense_dim,
            num_heads,
            dropout,
            num_blocks,
            positional_encoding,
            use_mask,
            max_len,
            temporal_vector,
        )
        self.ltv = LongTermView(embed_dim, msdcd)
        self.output_layer = nn.Linear(6 * embed_dim, 1)

    def forward(self, short_inputs: torch.Tensor, long_inputs: torch.Tensor):
        short_features = self.std(short_inputs)
        long_features = self.ltv(long_inputs)
        combined = torch.cat([short_features, long_features], dim=-1)
        return self.output_layer(combined).squeeze(-1)

    def predict_recursive(
        self,
        short_inputs: torch.Tensor,
        long_inputs: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        predictions = []
        for _ in range(steps):
            prediction = self(short_inputs, long_inputs)[:, -1].unsqueeze(1)
            predictions.append(prediction)
            next_value = prediction.unsqueeze(1)
            short_inputs = torch.cat([short_inputs[:, 1:, :], next_value], dim=1)
            long_inputs = torch.cat([long_inputs[:, 1:, :], next_value], dim=1)
        return torch.cat(predictions, dim=1)
