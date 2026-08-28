"""Test suite for the Keras3 module."""

import typing
import keras


class Model(keras.Model):  # type: ignore
  """Test suite for the Model component."""

  def __init__(self, in_features: int, out_features: int) -> None:
    """Initializes the Model instance."""
    super().__init__()
    self.linear: typing.Any = keras.layers.Dense(out_features, input_dim=in_features)  # type: ignore

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    return self.linear(x)
