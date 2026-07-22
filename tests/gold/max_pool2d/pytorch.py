"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class MaxPoolModel(nn.Module):
  """Test suite for the Max Pool Model component."""

  def __init__(self, kernel_size: int = 2, stride: int = 2):
    """Initializes the MaxPoolModel instance."""
    super().__init__()
    self.pool = nn.MaxPool2d(kernel_size, stride=stride)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.pool(x)
