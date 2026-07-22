"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
  """Test suite for the Embedding Model component."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance."""
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.emb(x)
