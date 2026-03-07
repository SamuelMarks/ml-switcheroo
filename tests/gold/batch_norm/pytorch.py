"""Module docstring."""

import torch
import torch.nn as nn


class BNModel(nn.Module):
  """Class docstring."""

  def __init__(self, num_features: int):
    """Function docstring."""
    super().__init__()
    self.bn = nn.BatchNorm2d(num_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    # PyTorch automatically updates running mean/var if self.training is True
    return self.bn(x)
