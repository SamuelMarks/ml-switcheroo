"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class ConvModel(nn.Module):
  """Class docstring."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Function docstring."""
    super().__init__()
    # MLX uses NHWC by default
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    return self.conv(x)
