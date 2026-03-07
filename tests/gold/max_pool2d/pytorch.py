"""Module docstring."""

import torch
import torch.nn as nn


class MaxPoolModel(nn.Module):
  """Class docstring."""

  def __init__(self, kernel_size: int = 2, stride: int = 2):
    """Function docstring."""
    super().__init__()
    self.pool = nn.MaxPool2d(kernel_size, stride=stride)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return self.pool(x)
