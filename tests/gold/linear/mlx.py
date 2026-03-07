"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class Model(nn.Module):
  """Class docstring."""

  def __init__(self, in_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    return self.linear(x)
