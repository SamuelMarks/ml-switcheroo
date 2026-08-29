"""Test suite for the Keras3 module."""

import typing

import keras


class GNModel(keras.Model):  # type: ignore
  """Docstring."""

  def __init__(self, num_groups: int, num_channels: int) -> None:
    """Initializes the GNModel instance."""
    super().__init__()
    self.gn: typing.Any = keras.layers.GroupNormalization(groups=num_groups)  # type: ignore

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    return self.gn(x)
