"""Module docstring."""

import torch
import torch.nn as nn


class SameConvModel(nn.Module):
  """Class docstring."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # PyTorch requires string 'same' padding to be used with stride=1
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
    # </SWITCHEROO_FAILED_TO_TRANS>

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return self.conv(x)
