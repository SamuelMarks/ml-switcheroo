"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class AttentionModel(nn.Module):
  """Class docstring."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Function docstring."""
    super().__init__()
    self.mha = nn.MultiHeadAttention(embed_dim, num_heads)

  def __call__(self, query: mx.array, key: mx.array, value: mx.array) -> mx.array:
    """Function docstring."""
    return self.mha(query, key, value)
