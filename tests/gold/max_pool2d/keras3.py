"""Test suite for the Keras3 module."""

import typing

import keras


class MaxPoolModel(keras.Model):  # type: ignore
  """Docstring."""

  def __init__(self, pool_size: int = 2, strides: int = 2) -> None:
    """Initializes the MaxPoolModel instance."""
    super().__init__()
    self.pool: typing.Any = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)  # type: ignore

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    return self.pool(x)
