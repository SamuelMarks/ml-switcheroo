"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class GNModel(nn.Module):
  """Test suite for the G N Model component."""

  def __init__(self, num_groups: int, num_channels: int):
    """Initializes the GNModel instance."""
    super().__init__()
    self.gn = nn.GroupNorm(num_groups, num_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.gn(x)
