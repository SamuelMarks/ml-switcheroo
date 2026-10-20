"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class AttentionModel:
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Test argument or fixture.
        num_heads: Test argument or fixture.
    """
    super().__init__()
    self.mha = nn.MultiHeadAttention(embed_dim, num_heads)

  def __call__(self, query: mx.array, key: mx.array, value: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        query: Test argument or fixture.
        key: Test argument or fixture.
        value: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.mha(query, key, value)
