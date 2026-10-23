"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class Model(nn.Module):
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance.

    Args:
        in_features: Number of input linear features.
        out_features: Number of output linear features.
    """
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.linear(x)
