"""Test suite for the Tensorflow module."""

import tensorflow as tf


class BNModel(tf.keras.Model):
  """Test suite for the B N Model component."""

  def __init__(self, num_features: int):
    """Initializes the BNModel instance."""
    super().__init__()
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
    """Helper to call."""
    return self.bn(x, training=training)
