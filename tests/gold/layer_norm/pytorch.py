"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class LayerNormModel(nn.Module):
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: The normalized shape (int) for   init  .
    """
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return self.ln(x)
