"""Module docstring."""

import mlx.core as mx
import mlx.nn as nn


def gelu_activation(x: mx.array, approximate: str = "none") -> mx.array:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # MLX has gelu and gelu_approx functions
  if approximate == "tanh":
    return nn.gelu_approx(x)
  return nn.gelu(x)
  # </SWITCHEROO_FAILED_TO_TRANS>
