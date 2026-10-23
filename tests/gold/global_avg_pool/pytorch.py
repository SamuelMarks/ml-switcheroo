"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class GAPModel(nn.Module):
  """Test helper class GAPModel."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return torch.mean(x, dim=(2, 3))
