"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class SameConvModel(nn.Module):
  """Class docstring."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # MLX lacks 'same' string, relies on explicit integer padding usually
    # equivalent for odd kernel: kernel_size // 2
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    return self.conv(x)
