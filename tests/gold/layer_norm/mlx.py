"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class LayerNormModel(nn.Module):
  """Test suite for the Layer Norm Model component."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance."""
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return self.ln(x)
