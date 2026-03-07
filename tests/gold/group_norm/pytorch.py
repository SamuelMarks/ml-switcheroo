"""Module docstring."""

import torch
import torch.nn as nn


class GNModel(nn.Module):
  """Class docstring."""

  def __init__(self, num_groups: int, num_channels: int):
    """Function docstring."""
    super().__init__()
    # PyTorch takes groups first
    self.gn = nn.GroupNorm(num_groups, num_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return self.gn(x)
