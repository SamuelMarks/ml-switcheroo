"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class FlattenModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance.

    Args:
        start_dim: Input `start_dim` parameter value (`int`).
    """
    super().__init__()
    self.flatten = nn.Flatten(start_dim=start_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `FlattenModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.flatten(x)
