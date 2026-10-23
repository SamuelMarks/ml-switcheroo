"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class GNModel(nn.Module):
  """Test helper class GNModel."""

  def __init__(self, num_groups: int, num_channels: int):
    """Initializes the GNModel instance.

    Args:
        num_groups: The num groups (int) for   init  .
        num_channels: The num channels (int) for   init  .
    """
    super().__init__()
    self.gn = nn.GroupNorm(num_groups, num_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.gn(x)
