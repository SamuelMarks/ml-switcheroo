"""Test suite for the Tensorflow module."""

import tensorflow as tf


class AttentionModel(tf.keras.Model):
  """Test suite for the Attention Model component."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance."""
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

  def call(self, query: tf.Tensor, value: tf.Tensor, key: tf.Tensor = None) -> tf.Tensor:
    """Helper to call."""
    return self.mha(query, value, key=key)
