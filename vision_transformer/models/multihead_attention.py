from typing import Optional

import torch
from torch import nn


class LinearMultiHeadAttention(nn.Module):
    """
    Class for linear transformation that splits a tensor into a specified number of heads for multi-head attention
    """
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int) -> None:
        """
        Initialization for the transformation

        Args:
            embed_dim: int - initial embedding dimensionality
            num_heads: int - number of heads
            head_dim: int - the resulting dimensionality of each head
        """
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_heads * head_dim)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformation

        Args:
             x: torch.Tensor - input tensor of shape [seq_len, batch_size, embed_dim]

        Returns:
            torch.Tensor - output tensor of shape [seq_len, batch_size, num_heads, head_dim]
        """
        x = self.linear(x)
        return x.view(*x.shape[:-1], self.num_heads, self.head_dim)


class MultiHeadAttention(nn.Module):
    """
    Class for multi-head attention
    """
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.query_transform = LinearMultiHeadAttention(embed_dim, num_heads, head_dim)
        self.key_transform = LinearMultiHeadAttention(embed_dim, num_heads, head_dim)
        self.value_transform = LinearMultiHeadAttention(embed_dim, num_heads, head_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention
        Args:
            query: torch.Tensor - query tensor of shape [seq_len, batch_size, embed_dim]
            key: torch.Tensor - key tensor of shape [seq_len, batch_size, embed_dim]
            value: torch.Tensor - value tensor of shape [seq_len, batch_size, embed_dim]
            mask: Optional[torch.Tensor] - mask tensor of shape [seq_len, seq_len, batch_size]. mask[i, j] indicates
                                           whether i-th token has access to j-th token

        Returns:
            torch.Tensor - output tensor of shape [seq_len, batch_size, num_heads, head_dim]
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)

        query = self.query_transform(query)
        key = self.key_transform(key)
        value = self.value_transform(value)
        return value
