"""Test suite for the Pytorch module."""

import typing

import torch
import torch.nn as nn


class GNModel(nn.Module):
  """Test helper class GNModel."""

  def __init__(self, num_groups: int, num_channels: int):
    """Initializes the GNModel instance.

    Args:
        num_groups: Test argument or fixture.
        num_channels: Test argument or fixture.
    """
    super().__init__()
    self.gn = nn.GroupNorm(num_groups, num_channels)

  def forward(self, x: torch.Tensor) -> typing.Any:
    """Helper to forward.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.gn(x)
