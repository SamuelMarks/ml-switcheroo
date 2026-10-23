"""Test suite for the Tensorflow module."""

import tensorflow as tf


class AttentionModel:
  """Test helper class AttentionModel."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: The embed dim (int) for   init  .
        num_heads: The num heads (int) for   init  .
    """
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

  def call(self, query: tf.Tensor, value: tf.Tensor, key: tf.Tensor = None) -> tf.Tensor:
    """Helper to call.

    Args:
        query: The query (tf.Tensor) for call.
        value: The value (tf.Tensor) for call.
        key: The key (tf.Tensor) for call.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.mha(query, value, key=key)
