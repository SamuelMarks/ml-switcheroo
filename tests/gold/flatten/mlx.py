"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class FlattenModel(nn.Module):
  """Class docstring."""

  def __init__(self, start_dim: int = 1):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # MLX does not have a Flatten layer in nn usually, uses functional flattening
    self.start_dim = start_dim
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    return mx.flatten(x, start_axis=self.start_dim)
