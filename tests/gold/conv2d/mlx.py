"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class ConvModel(nn.Module):
  """Test suite for the Conv Model component."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the ConvModel instance."""
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return self.conv(x)
