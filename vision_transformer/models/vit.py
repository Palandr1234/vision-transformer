import torch
from torch import nn

from vision_transformer.models.transformer_layer import TransformerLayer


class PatchEmbedding(nn.Module):
    """
    Class for patch embedding for the original paper
    """
    def __init__(self, embed_dim: int, patch_size: int, in_channels: int = 3) -> None:
        """
        Initialization of patch embedding

        Args:
            embed_dim: int - embedding dimensionality
            patch_size: int - size of each patch
            in_channels: int - number of input channels
        """
        super().__init__()
        # creating such a convolutional kernel is equivalent to splitting the image into patches and applying linear
        # transformation separately
        self.embed_dim = embed_dim
        self.embedding = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding class

        Args:
            x: torch.Tensor - input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            torch.Tensor - patch embeddings of shape [num_patches, batch_size, embed_dim]
        """
        batch_size = x.size(0)
        x = self.embedding(x)
        return x.permute(2, 3, 0, 1).view(-1, batch_size, self.embed_dim)


class VisionTransformer(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, patch_size: int, num_heads: int, hidden_dim1: int,
                 hidden_dim2: int, num_layers: int, in_channels: int = 3, max_num_patches: int = 5000,
                 dropout_prob: float = 0.1) -> None:
        """
        Initialization of vision transformer

        Args:
            embed_dim: int - embedding dimensionality
            patch_size: int - size of each patch
            in_channels: int - number of input channels
            max_num_patches: int - maximum number of patches
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(embed_dim, patch_size, in_channels)
        self.positional_embedding = nn.Parameter(torch.randn(max_num_patches, 1, embed_dim), requires_grad=True)
        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.transformer_layers = nn.ModuleList([TransformerLayer(embed_dim, num_heads, hidden_dim1, dropout_prob)
                                                 for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm([embed_dim])
        self.linear1 = nn.Linear(embed_dim, hidden_dim2)
        self.activation = nn.GeLU()
        self.linear2 = nn.Linear(hidden_dim2, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding class

        Args:
            x: torch.Tensor - input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            torch.Tensor - patch embeddings of shape [batch_size, num_classes]
        """
        x = self.patch_embedding(x)
        x = x + self.positional_embedding[:x.shape[0]]
        cls_token_emb = self.cls_token_embedding.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])
        for layer in self.transformer_layers:
            x = layer(x, mask=None)

        #  Get the transformer output of the [CLS] token
        x = self.layer_norm(x[0])
        x = self.activation(self.linear1(x))
        return self.softmax(self.linear2(x))
