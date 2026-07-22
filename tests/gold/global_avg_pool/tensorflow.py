"""Test suite for the Tensorflow module."""

import tensorflow as tf


class GAPModel(tf.keras.Model):
  """Test suite for the G A P Model component."""

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return tf.math.reduce_mean(x, axis=(1, 2))
