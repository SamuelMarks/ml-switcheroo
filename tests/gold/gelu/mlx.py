"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


def gelu_activation(x: mx.array, approximate: mx.array = "none") -> mx.array:
  """Execute `nn.gelu` to support gelu activation in MLX tests.

  Args:
      x: Primary input tensor or array operand.
      approximate: Input `approximate` parameter value (`mx.array`).

  Returns:
      Resulting `mx.array` from gelu activation.
  """
  if approximate == "tanh":
    return nn.gelu_approx(x)
  return nn.gelu(x)
