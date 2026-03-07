"""Module docstring."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
  """Class docstring."""

  def __init__(self, channels: int):
    """Function docstring."""
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    out = self.relu(out)
    return out
