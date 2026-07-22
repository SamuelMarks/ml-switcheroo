"""Test suite for the Tensorflow module."""

import tensorflow as tf


class FlattenModel(tf.keras.Model):
  """Test suite for the Flatten Model component."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance."""
    super().__init__()
    self.flatten = tf.keras.layers.Flatten()

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return self.flatten(x)
