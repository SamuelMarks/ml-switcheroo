"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class DropoutModel(nn.Module):
  """Test suite for the Dropout Model component."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance."""
    super().__init__()
    self.dropout = nn.Dropout(p)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return self.dropout(x)
