"""Test suite for the Tensorflow module."""

import tensorflow as tf


class DropoutModel(tf.keras.Model):
  """Test suite for the Dropout Model component."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance."""
    super().__init__()
    self.dropout = tf.keras.layers.Dropout(p)

  def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
    """Helper to call."""
    return self.dropout(x, training=training)
