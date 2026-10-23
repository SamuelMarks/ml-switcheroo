"""Test suite for the Keras3 module."""

import keras


class AttentionModel:
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int) -> None:
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: The embed dim (int) for   init  .
        num_heads: The num heads (int) for   init  .
    """
    super().__init__()
    self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

  def call(
    self, query: keras.KerasTensor, value: keras.KerasTensor, key: keras.KerasTensor | None = None
  ) -> keras.KerasTensor:
    """Helper to call.

    Args:
        query: The query (keras.KerasTensor) for call.
        value: The value (keras.KerasTensor) for call.
        key: The key (keras.KerasTensor | None) for call.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.mha(query, value, key=key)
