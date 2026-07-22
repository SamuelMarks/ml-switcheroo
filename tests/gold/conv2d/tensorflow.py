"""Test suite for the Tensorflow module."""

import tensorflow as tf


class ConvModel(tf.keras.Model):
  """Test suite for the Conv Model component."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the ConvModel instance."""
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size, input_shape=(None, None, in_channels))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call."""
    return self.conv(x)
