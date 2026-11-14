"""Test suite for the Tensorflow module."""

import tensorflow as tf


class AttentionModel:
  """Test double `AttentionModel` implementing `call` for tensorflow tests."""

  def __init__(self, embed_dim: int, num_heads: int):
    """Initializes the AttentionModel instance.

    Args:
        embed_dim: Input `embed_dim` parameter value (`int`).
        num_heads: Input `num_heads` parameter value (`int`).
    """
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

  def call(self, query: tf.Tensor, value: tf.Tensor, key: tf.Tensor = None) -> tf.Tensor:
    """Execute `mha` to support call in tensorflow tests.

    Args:
        query: Input `query` parameter value (`tf.Tensor`).
        value: Input `value` parameter value (`tf.Tensor`).
        key: Input `key` parameter value (`tf.Tensor`).

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.mha(query, value, key=key)
