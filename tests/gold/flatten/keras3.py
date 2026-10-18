"""Test suite for the Keras3 module."""

import typing

import keras


class FlattenModel:
  """Test helper class FlattenModel."""

  def __init__(self, start_dim: int = 1) -> None:
    """Initializes the FlattenModel instance.

    Args:
        start_dim: Test argument or fixture.
    """
    super().__init__()
    self.flatten: typing.Any = keras.layers.Flatten()

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.flatten(x)
