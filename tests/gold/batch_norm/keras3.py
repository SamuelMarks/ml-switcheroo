"""Test suite for the Keras3 module."""

import typing

import keras


class BNModel:
  """Test helper class BNModel."""

  def __init__(self, num_features: int) -> None:
    """Initializes the BNModel instance.

    Args:
        num_features: Test argument or fixture.
    """
    super().__init__()
    self.bn = keras.layers.BatchNormalization()

  def call(self, x: typing.Any, training: typing.Any = None) -> typing.Any:
    """Helper to call.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.bn(x, training=training)
