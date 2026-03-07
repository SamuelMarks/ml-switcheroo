"""Module docstring."""

import torch
import torch.nn as nn


class Model(nn.Module):
  """Class docstring."""

  def __init__(self, in_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return self.linear(x)
