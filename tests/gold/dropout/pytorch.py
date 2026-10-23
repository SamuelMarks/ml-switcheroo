"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class DropoutModel(nn.Module):
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance.

    Args:
        p: The p (float) for   init  .
    """
    super().__init__()
    self.dropout = nn.Dropout(p)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.dropout(x)
