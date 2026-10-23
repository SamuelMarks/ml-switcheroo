"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class MLP(nn.Module):
  """Test helper class MLP."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Initializes the MLP instance.

    Args:
        in_features: Number of input linear features.
        hidden_features: The hidden features (int) for   init  .
        out_features: Number of output linear features.
    """
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
