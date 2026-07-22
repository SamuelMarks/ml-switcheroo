"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class MaxPoolModel(nn.Module):
  """Test suite for the Max Pool Model component."""

  def __init__(self, kernel_size: int = 2, stride: int = 2):
    """Initializes the MaxPoolModel instance."""
    super().__init__()
    self.pool = nn.MaxPool2d(kernel_size, stride=stride)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return self.pool(x)
