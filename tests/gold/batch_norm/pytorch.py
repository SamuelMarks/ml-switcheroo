"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class BNModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, num_features: int):
    """Initializes the BNModel instance.

    Args:
        num_features: Input `num_features` parameter value (`int`).
    """
    super().__init__()
    self.bn = nn.BatchNorm2d(num_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `BNModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.bn(x)
