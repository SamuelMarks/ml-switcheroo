"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class WNModel(nn.Module):
  """Test helper class WNModel."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the WNModel instance.

    Args:
        in_features: Number of input linear features.
        out_features: Number of output linear features.
    """
    super().__init__()
    self.linear = weight_norm(nn.Linear(in_features, out_features))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.linear(x)
