"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class Model(nn.Module):
  """Docstring."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance."""
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return self.linear(x)
