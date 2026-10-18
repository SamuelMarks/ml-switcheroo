"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class AttentionModel:
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: typing.Any, num_heads: typing.Any):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Test argument or fixture.
        num_heads: Test argument or fixture.
    """
    super().__init__()
    self.mha = nn.MultiHeadAttention(embed_dim, num_heads)

  def __call__(self, query: typing.Any, key: typing.Any, value: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        query: Test argument or fixture.
        key: Test argument or fixture.
        value: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.mha(query, key, value)
