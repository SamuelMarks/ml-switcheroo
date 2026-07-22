"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class BNModel(nn.Module):
  """Test suite for the B N Model component."""

  def __init__(self, num_features: int):
    """Initializes the BNModel instance."""
    super().__init__()
    self.bn = nn.BatchNorm(num_features)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return self.bn(x)
