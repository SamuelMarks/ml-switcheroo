"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


class LayerNormModel(nn.Module):
  """Test suite for the Layer Norm Model component."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance."""
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Helper to forward."""
    return self.ln(x)
