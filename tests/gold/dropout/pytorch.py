"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class DropoutModel(nn.Module):
  """Test suite for the Dropout Model component."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance."""
    super().__init__()
    self.dropout = nn.Dropout(p)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.dropout(x)
