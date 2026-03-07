"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


class GAPModel(nn.Module):
  """Class docstring."""

  def __call__(self, x: mx.array) -> mx.array:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # MLX assumes NHWC conventionally
    return mx.mean(x, axis=(1, 2))
    # </SWITCHEROO_FAILED_TO_TRANS>
