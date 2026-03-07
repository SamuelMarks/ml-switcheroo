"""Module docstring."""

import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
  """Class docstring."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Function docstring."""
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return self.emb(x)
