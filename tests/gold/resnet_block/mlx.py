"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class ResidualBlock(nn.Module):
  """Test suite for the Residual Block component."""

  def __init__(self, channels: int):
    """Initializes the ResidualBlock instance."""
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm(channels)
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm(channels)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = nn.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + residual
    out = nn.relu(out)
    return out
