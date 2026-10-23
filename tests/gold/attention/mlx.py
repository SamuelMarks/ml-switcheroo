"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


class AttentionModel:
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: The embed dim (int) for   init  .
        num_heads: The num heads (int) for   init  .
    """
    super().__init__()
    self.mha = nn.MultiHeadAttention(embed_dim, num_heads)

  def __call__(self, query: mx.array, key: mx.array, value: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        query: The query (mx.array) for   call  .
        key: The key (mx.array) for   call  .
        value: The value (mx.array) for   call  .

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.mha(query, key, value)
