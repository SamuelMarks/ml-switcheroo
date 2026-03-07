"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class LayerNormModel(nn.Module):
  """Class docstring."""

  def __init__(self, normalized_shape: int):
    """Function docstring."""
    super().__init__()
    self.ln = nn.LayerNorm(normalized_shape)

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    return self.ln(x)
