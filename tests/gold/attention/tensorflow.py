"""Module docstring."""

import tensorflow as tf


class AttentionModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Function docstring."""
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

  def call(self, query: tf.Tensor, value: tf.Tensor, key: tf.Tensor = None) -> tf.Tensor:
    """Function docstring."""
    return self.mha(query, value, key=key)
