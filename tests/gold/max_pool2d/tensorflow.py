"""Test suite for the Tensorflow module."""

import tensorflow as tf


class MaxPoolModel(tf.keras.Model):
  """Test suite for the Max Pool Model component."""

  def __init__(self, pool_size: int = 2, strides: int = 2):
    """Initializes the MaxPoolModel instance."""
    super().__init__()
    self.pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return self.pool(x)
