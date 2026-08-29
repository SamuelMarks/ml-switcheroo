"""Test suite for the Tensorflow module."""

import typing

import tensorflow as tf


class ResidualBlock(tf.keras.Model):  # type: ignore
  """Docstring."""

  def __init__(self, channels: int) -> None:
    """Initializes the ResidualBlock instance."""
    super().__init__()
    self.conv1: typing.Any = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn1: typing.Any = tf.keras.layers.BatchNormalization()
    self.conv2: typing.Any = tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn2: typing.Any = tf.keras.layers.BatchNormalization()

  def call(self, x: typing.Any, training: typing.Optional[bool] = None) -> typing.Any:
    """Helper to call."""
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = tf.nn.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    out = out + residual
    out = tf.nn.relu(out)
    return out
