"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class BNModel(nn.Module):
  """Class docstring."""

  def __init__(self, num_features: int):
    """Function docstring."""
    super().__init__()
    self.bn = nn.BatchNorm(num_features)

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    return self.bn(x)
