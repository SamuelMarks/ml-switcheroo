"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class FlattenModel(nn.Module):
  """Test suite for the Flatten Model component."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance."""
    super().__init__()
    self.start_dim = start_dim

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return mx.flatten(x, start_axis=self.start_dim)
