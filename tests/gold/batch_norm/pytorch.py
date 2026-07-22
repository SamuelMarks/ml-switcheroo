"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class BNModel(nn.Module):
  """Test suite for the B N Model component."""

  def __init__(self, num_features: int):
    """Initializes the BNModel instance."""
    super().__init__()
    self.bn = nn.BatchNorm2d(num_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.bn(x)
