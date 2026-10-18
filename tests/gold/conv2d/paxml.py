"""Test suite for the Paxml module."""

import typing

from praxis.layers import convolutions


class ConvModel:
  """Test helper class ConvModel."""

  in_channels: typing.Any = 0
  out_channels: typing.Any = 0
  kernel_size: typing.Any = 3

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child(
      "conv",
      convolutions.Conv2D.HParams(filter_shape=(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)),
    )

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.conv(x)
