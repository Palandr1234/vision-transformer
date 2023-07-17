import torch
from torch import nn

from vision_transformer.models.multihead_attention import MultiHeadAttention


class TransformerLayer(nn.Module):
    """
    Class for transformer layer
    """
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout_prob: float = 0.1) -> None:
        """
        Initialization for transformer layer

        Args:
            embed_dim: int - embedding dimensionality
            num_heads: int - number of heads
            dropout_prob - probability of dropout being applied
        """
        super().__init__()
        self.norm_self_attention = nn.LayerNorm([embed_dim])
        self.norm_feed_forward = nn.LayerNorm([embed_dim])
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GeLU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: torch.Tensor - input tensor of shape [seq_len, batch_size, embed_dim]
            mask: torch.Tensor - mask tensor of shape [seq_len, seq_len, batch_size]. mask[i, j] indicates
                                 whether i-th token has access to j-th token

        Returns:
            torch.Tensor - output tensor of shape [seq_len, batch_size, embed_dim]
        """
        x1 = self.norm_self_attention(x)
        x = x + self.dropout(self.self_attention(query=x1, key=x1, value=x1, mask=mask))

        x2 = self.norm_feed_forward(x)
        x2 = self.activation(self.linear1(x2))
        x2 = self.dropout(x2)
        return x + self.dropout(self.linear2(x2))
