"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class GAPModel(nn.Module):
  """Test suite for the G A P Model component."""

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance."""
    return mx.mean(x, axis=(1, 2))
