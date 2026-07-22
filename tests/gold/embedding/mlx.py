"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class EmbeddingModel(nn.Module):
  """Test suite for the Embedding Model component."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Initializes the EmbeddingModel instance."""
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return self.emb(x)
