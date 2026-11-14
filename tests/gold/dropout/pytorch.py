"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class DropoutModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance.

    Args:
        p: Input `p` parameter value (`float`).
    """
    super().__init__()
    self.dropout = nn.Dropout(p)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `DropoutModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.dropout(x)
