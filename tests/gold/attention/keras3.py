"""Test suite for the Keras3 module."""

import keras


class AttentionModel(keras.Model):
  """Test suite for the Attention Model component."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance."""
    super().__init__()
    self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

  def call(self, query, value, key=None):
    """Helper to call."""
    return self.mha(query, value, key=key)
