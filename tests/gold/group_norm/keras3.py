"""Test suite for the Keras3 module."""

import typing

import keras


class GNModel:
  """Test helper class GNModel."""

  def __init__(self, num_groups: int, num_channels: int) -> None:
    """Initializes the GNModel instance.

    Args:
        num_groups: Test argument or fixture.
        num_channels: Test argument or fixture.
    """
    super().__init__()
    self.gn: typing.Any = keras.layers.GroupNormalization(groups=num_groups)

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.gn(x)
