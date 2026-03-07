"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class EmbeddingModel(nn.Module):
  """Class docstring."""

  def __init__(self, num_embeddings: int, embedding_dim: int):
    """Function docstring."""
    super().__init__()
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    return self.emb(x)
