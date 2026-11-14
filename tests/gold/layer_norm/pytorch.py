"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class LayerNormModel(nn.Module):
  """Stub `nn.Module` subclass implementing `forward` for pytorch tests."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Input `normalized_shape` parameter value (`int`).
    """
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward pass of `LayerNormModel` on input tensors.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.ln(x)
