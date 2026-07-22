"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class WNModel(nn.Module):
  """Test suite for the W N Model component."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the WNModel instance."""
    super().__init__()
    self.linear = weight_norm(nn.Linear(in_features, out_features))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.linear(x)
