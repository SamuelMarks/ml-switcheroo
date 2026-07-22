"""Test suite for the Tensorflow module."""

import tensorflow as tf
import tensorflow_addons as tfa


class WNModel(tf.keras.Model):
  """Test suite for the W N Model component."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the WNModel instance."""
    super().__init__()
    self.linear = tfa.layers.WeightNormalization(tf.keras.layers.Dense(out_features))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return self.linear(x)
