"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class BNModel(nn.Module):
  """Test helper class BNModel."""

  def __init__(self, num_features: int):
    """Initializes the BNModel instance.

    Args:
        num_features: The num features (int) for   init  .
    """
    super().__init__()
    self.bn = nn.BatchNorm2d(num_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.bn(x)
