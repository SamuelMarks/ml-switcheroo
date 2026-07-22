"""Test suite for the Tensorflow module."""

import tensorflow as tf


class Model(tf.keras.Model):
  """Test suite for the Model component."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance."""
    super().__init__()
    self.linear = tf.keras.layers.Dense(out_features, input_shape=(in_features,))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return self.linear(x)
