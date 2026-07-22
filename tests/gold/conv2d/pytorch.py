"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class ConvModel(nn.Module):
  """Test suite for the Conv Model component."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the ConvModel instance."""
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.conv(x)
