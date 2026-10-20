"""Test suite for the Tensorflow module."""

import typing

import tensorflow as tf


class ResidualBlock:
  """Test helper class ResidualBlock."""

  def __init__(self, channels: int) -> None:
    """Initializes the ResidualBlock instance.

    Args:
        channels: Test argument or fixture.
    """
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn2 = tf.keras.layers.BatchNormalization()

  def call(self, x: tf.Tensor, training: typing.Optional[bool] = None) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = tf.nn.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    out = out + residual
    out = tf.nn.relu(out)
    return out
