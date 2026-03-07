"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class MaxPoolModel(nn.Module):
  """Class docstring."""

  def __init__(self, kernel_size: int = 2, stride: int = 2):
    """Function docstring."""
    super().__init__()
    self.pool = nn.MaxPool2d(kernel_size, stride=stride)

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    return self.pool(x)
