"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class AttentionModel:
  """Sentinel `AttentionModel` fixture class used in MLX tests."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Input `embed_dim` parameter value (`int`).
        num_heads: Input `num_heads` parameter value (`int`).
    """
    super().__init__()
    self.mha = nn.MultiHeadAttention(embed_dim, num_heads)

  def __call__(self, query: mx.array, key: mx.array, value: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        query: Input `query` parameter value (`mx.array`).
        key: Input `key` parameter value (`mx.array`).
        value: Input `value` parameter value (`mx.array`).

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.mha(query, key, value)
