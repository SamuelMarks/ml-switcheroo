"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class GNModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, num_groups: int, num_channels: int):
    """Initializes the GNModel instance.

    Args:
        num_groups: Input `num_groups` parameter value (`int`).
        num_channels: Input `num_channels` parameter value (`int`).
    """
    super().__init__()
    self.gn = nn.GroupNorm(num_groups, num_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `GNModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.gn(x)
