import torch
from torch import nn


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
    def __init__(self, embed_dim: int, patch_size: int, in_channels: int = 3):
        """
        Initialization of vision transformer

        Args:
            embed_dim: int - embedding dimensionality
            patch_size: int - size of each patch
            in_channels: int - number of input channels
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(embed_dim, patch_size, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding class

        Args:
            x: torch.Tensor - input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            torch.Tensor - patch embeddings of shape [num_patches, batch_size, embed_dim]
        """
        return self.patch_embedding(x)
