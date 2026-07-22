"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class FlattenModel(nn.Module):
  """Test suite for the Flatten Model component."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance."""
    super().__init__()
    self.flatten = nn.Flatten(start_dim=start_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.flatten(x)
