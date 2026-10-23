"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class FlattenModel(nn.Module):
  """Test helper class FlattenModel."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance.

    Args:
        start_dim: The start dim (int) for   init  .
    """
    super().__init__()
    self.flatten = nn.Flatten(start_dim=start_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.flatten(x)
