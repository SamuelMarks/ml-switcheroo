"""Module docstring."""

import torch
import torch.nn as nn


class MLP(nn.Module):
  """Class docstring."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
