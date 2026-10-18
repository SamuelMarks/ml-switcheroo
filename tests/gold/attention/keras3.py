"""Test suite for the Keras3 module."""

import typing

import keras


class AttentionModel:
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int) -> None:
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Test argument or fixture.
        num_heads: Test argument or fixture.
    """
    super().__init__()
    self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

  def call(self, query: typing.Any, value: typing.Any, key: typing.Any = None) -> typing.Any:
    """Helper to call.

    Args:
        query: Test argument or fixture.
        value: Test argument or fixture.
        key: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.mha(query, value, key=key)
