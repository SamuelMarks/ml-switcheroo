"""Module docstring."""

import torch
import torch.nn as nn


class FlattenModel(nn.Module):
  """Class docstring."""

  def __init__(self, start_dim: int = 1):
    """Function docstring."""
    super().__init__()
    # PyTorch keeps batch dimension by default (start_dim=1)
    self.flatten = nn.Flatten(start_dim=start_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return self.flatten(x)
