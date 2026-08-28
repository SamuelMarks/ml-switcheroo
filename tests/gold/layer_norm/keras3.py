"""Test suite for the Keras3 module."""

import typing
import keras


class LayerNormModel(keras.Model):  # type: ignore
  """Test suite for the Layer Norm Model component."""

  def __init__(self, normalized_shape: int) -> None:
    """Initializes the LayerNormModel instance."""
    super().__init__()
    self.ln: typing.Any = keras.layers.LayerNormalization(axis=-1)  # type: ignore

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    return self.ln(x)
