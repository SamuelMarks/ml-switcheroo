"""Test suite for the Tensorflow module."""

import tensorflow as tf


class SameConvModel(tf.keras.Model):
  """Test suite for the Same Conv Model component."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the SameConvModel instance."""
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size, padding="same")

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return self.conv(x)
