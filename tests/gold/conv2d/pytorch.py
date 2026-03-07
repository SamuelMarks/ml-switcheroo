"""Module docstring."""

import torch
import torch.nn as nn


class ConvModel(nn.Module):
  """Class docstring."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Function docstring."""
    super().__init__()
    # PyTorch uses NCHW by default
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return self.conv(x)
