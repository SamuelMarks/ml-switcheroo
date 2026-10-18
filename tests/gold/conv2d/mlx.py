"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class ConvModel:
  """Test helper class ConvModel."""

  def __init__(self, in_channels: typing.Any, out_channels: typing.Any, kernel_size: typing.Any = 3):
    """Initializes the ConvModel instance.

    Args:
        in_channels: Test argument or fixture.
        out_channels: Test argument or fixture.
        kernel_size: Test argument or fixture.
    """
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.conv(x)
