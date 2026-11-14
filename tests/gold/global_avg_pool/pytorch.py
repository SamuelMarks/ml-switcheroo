"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class GAPModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `GAPModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return torch.mean(x, dim=(2, 3))
