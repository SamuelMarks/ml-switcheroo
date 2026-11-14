"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, channels: int):
    """Initializes the ResidualBlock instance.

    Args:
        channels: Input `channels` parameter value (`int`).
    """
    super().__init__()
    self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `ResidualBlock` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    return self.relu(out)
