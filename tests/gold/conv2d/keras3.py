"""Test suite for the Keras3 module."""

import typing

import keras


class ConvModel(keras.Model):  # type: ignore
  """Docstring."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
    """Initializes the ConvModel instance."""
    super().__init__()
    self.conv: typing.Any = keras.layers.Conv2D(out_channels, kernel_size, input_shape=(None, None, in_channels))  # type: ignore

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    return self.conv(x)
