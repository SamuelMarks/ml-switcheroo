"""Test suite for the Keras3 module."""

import typing

import keras


class ResidualBlock(keras.Model):  # type: ignore
  """Docstring."""

  def __init__(self, channels: int) -> None:
    """Initializes the ResidualBlock instance."""
    super().__init__()
    self.conv1: typing.Any = keras.layers.Conv2D(channels, kernel_size=3, padding="same")  # type: ignore
    self.bn1: typing.Any = keras.layers.BatchNormalization()  # type: ignore
    self.conv2: typing.Any = keras.layers.Conv2D(channels, kernel_size=3, padding="same")  # type: ignore
    self.bn2: typing.Any = keras.layers.BatchNormalization()  # type: ignore

  def call(self, x: typing.Any, training: typing.Optional[bool] = None) -> typing.Any:
    """Helper to call."""
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = keras.activations.relu(out)  # type: ignore
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    out = out + residual
    out = keras.activations.relu(out)  # type: ignore
    return out
