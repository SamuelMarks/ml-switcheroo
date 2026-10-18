"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class ResidualBlock:
  """Test helper class ResidualBlock."""

  def __init__(self, channels: typing.Any):
    """Initializes the ResidualBlock instance.

    Args:
        channels: Test argument or fixture.
    """
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm(channels)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm(channels)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    residual: typing.Any = x
    out: typing.Any = self.conv1(x)
    out = self.bn1(out)
    out = nn.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + residual
    out = nn.relu(out)
    return out
