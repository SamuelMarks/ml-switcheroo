"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class MLP(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Initializes the MLP instance.

    Args:
        in_features: Number of input linear features.
        hidden_features: Input `hidden_features` parameter value (`int`).
        out_features: Number of output linear features.
    """
    super().__init__()
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `MLP` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    x = self.fc1(x)
    x = self.relu(x)
    return self.fc2(x)
