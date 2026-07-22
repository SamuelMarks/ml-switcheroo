"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class AttentionModel(nn.Module):
  """Test suite for the Attention Model component."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance."""
    super().__init__()
    self.mha = nn.MultiHeadAttention(embed_dim, num_heads)

  def __call__(self, query: mx.array, key: mx.array, value: mx.array) -> mx.array:
    """Executes the callable instance."""
    return self.mha(query, key, value)
