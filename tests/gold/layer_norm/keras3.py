"""Test suite for the Keras3 module."""

import typing

import keras


class LayerNormModel:
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int) -> None:
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Test argument or fixture.
    """
    super().__init__()
    self.ln: typing.Any = keras.layers.LayerNormalization(axis=-1)

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.ln(x)
