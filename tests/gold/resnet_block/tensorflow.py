"""Test suite for the Tensorflow module."""

import tensorflow as tf


class ResidualBlock:
  """Test double `ResidualBlock` implementing `call` for tensorflow tests."""

  def __init__(self, channels: int) -> None:
    """Initializes the ResidualBlock instance.

    Args:
        channels: Input `channels` parameter value (`int`).
    """
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn2 = tf.keras.layers.BatchNormalization()

  def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
    """Execute `conv1` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.
        training: Input `training` parameter value (`bool | None`).

    Returns:
        Resulting `tf.Tensor` from call.
    """
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = tf.nn.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    out = out + residual
    return tf.nn.relu(out)
