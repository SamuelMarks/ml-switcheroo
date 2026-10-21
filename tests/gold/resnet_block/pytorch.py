"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
  """Test helper class ResidualBlock."""

  def __init__(self, channels: int):
    """Initializes the ResidualBlock instance.

    Args:
        channels: Test argument or fixture.
    """
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    out = self.relu(out)
    return out
