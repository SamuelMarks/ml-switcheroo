"""Module docstring."""

import tensorflow as tf


class SameConvModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Function docstring."""
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size, padding="same")

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    return self.conv(x)
