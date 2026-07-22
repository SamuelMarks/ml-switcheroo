"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class MLP(nn.Module):
  """Test suite for the M L P component."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Initializes the MLP instance."""
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
