"""Test suite for the Keras3 module."""

import typing
import keras


class FlattenModel(keras.Model):  # type: ignore
  """Test suite for the Flatten Model component."""

  def __init__(self, start_dim: int = 1) -> None:
    """Initializes the FlattenModel instance."""
    super().__init__()
    self.flatten: typing.Any = keras.layers.Flatten()  # type: ignore

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    return self.flatten(x)
