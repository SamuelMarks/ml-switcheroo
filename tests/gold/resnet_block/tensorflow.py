"""Module docstring."""

import tensorflow as tf


class ResidualBlock(tf.keras.Model):
  """Class docstring."""

  def __init__(self, channels: int):
    """Function docstring."""
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn2 = tf.keras.layers.BatchNormalization()

  def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
    """Function docstring."""
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = tf.nn.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    out = out + residual
    out = tf.nn.relu(out)
    return out
