"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class Model(nn.Module):
  """Test suite for the Model component."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance."""
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.linear(x)
