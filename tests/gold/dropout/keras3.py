"""Test suite for the Keras3 module."""

import typing
import keras


class DropoutModel(keras.Model):  # type: ignore
  """Test suite for the Dropout Model component."""

  def __init__(self, p: float = 0.5) -> None:
    """Initializes the DropoutModel instance."""
    super().__init__()
    self.dropout: typing.Any = keras.layers.Dropout(p)  # type: ignore

  def call(self, x: typing.Any, training: typing.Optional[bool] = None) -> typing.Any:
    """Helper to call."""
    return self.dropout(x, training=training)
