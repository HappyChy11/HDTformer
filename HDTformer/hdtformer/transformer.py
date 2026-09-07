import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.encoding[:, : inputs.size(1), :]


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dense_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attention, _ = self.attention(inputs, inputs, inputs, need_weights=False)
        outputs = self.norm1(inputs + self.attention_dropout(attention))
        return self.norm2(
            outputs + self.feed_forward_dropout(self.feed_forward(outputs))
        )


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dense_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, embed_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention, _ = self.self_attention(
            inputs, inputs, inputs, attn_mask=mask, need_weights=False
        )
        outputs = self.norm1(inputs + self.dropout1(attention))
        cross_attention, _ = self.cross_attention(
            outputs, encoder_outputs, encoder_outputs, need_weights=False
        )
        outputs = self.norm2(outputs + self.dropout2(cross_attention))
        return self.norm3(outputs + self.dropout3(self.feed_forward(outputs)))
