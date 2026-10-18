"""Test suite for the Keras3 module."""

import typing

import keras


class SameConvModel:
  """Test helper class SameConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
    """Initializes the SameConvModel instance.

    Args:
        in_channels: Test argument or fixture.
        out_channels: Test argument or fixture.
        kernel_size: Test argument or fixture.
    """
    super().__init__()
    self.conv: typing.Any = keras.layers.Conv2D(out_channels, kernel_size, padding="same")

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.conv(x)
