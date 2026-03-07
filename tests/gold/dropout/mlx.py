"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class DropoutModel(nn.Module):
  """Class docstring."""

  def __init__(self, p: float = 0.5):
    """Function docstring."""
    super().__init__()
    self.dropout = nn.Dropout(p)

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    # MLX relies on the global self.training flag implicitly in __call__ if not provided
    return self.dropout(x)
