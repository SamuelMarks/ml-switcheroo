"""Test suite for the Keras3 module."""

import keras


class AttentionModel:
  """Test double `AttentionModel` implementing `call` for keras3 tests."""

  def __init__(self, embed_dim: int, num_heads: int) -> None:
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Input `embed_dim` parameter value (`int`).
        num_heads: Input `num_heads` parameter value (`int`).
    """
    super().__init__()
    self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

  def call(
    self, query: keras.KerasTensor, value: keras.KerasTensor, key: keras.KerasTensor | None = None
  ) -> keras.KerasTensor:
    """Execute `mha` to support call in keras3 tests.

    Args:
        query: Input `query` parameter value (`keras.KerasTensor`).
        value: Input `value` parameter value (`keras.KerasTensor`).
        key: Input `key` parameter value (`keras.KerasTensor | None`).

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.mha(query, value, key=key)
