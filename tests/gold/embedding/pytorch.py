"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
  """Test helper class EmbeddingModel."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance.

    Args:
        num_embeddings: The num embeddings (int) for   init  .
        embedding_dim: The embedding dim (int) for   init  .
    """
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.emb(x)
