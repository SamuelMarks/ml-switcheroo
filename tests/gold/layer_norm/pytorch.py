"""Module docstring."""

import torch
import torch.nn as nn


class LayerNormModel(nn.Module):
  """Class docstring."""

  def __init__(self, normalized_shape: int):
    """Function docstring."""
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Function docstring."""
    return self.ln(x)
