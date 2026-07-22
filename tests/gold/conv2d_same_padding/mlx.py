"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class SameConvModel(nn.Module):
  """Test suite for the Same Conv Model component."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the SameConvModel instance."""
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return self.conv(x)
