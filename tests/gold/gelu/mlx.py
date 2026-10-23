"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


def gelu_activation(x: mx.array, approximate: mx.array = "none") -> mx.array:
  """Helper to gelu activation.

  Args:
      x: Primary input tensor or array operand.
      approximate: The approximate (mx.array) for gelu activation.

  Returns:
      Resulting `mx.array` from gelu activation.
  """
  if approximate == "tanh":
    return nn.gelu_approx(x)
  return nn.gelu(x)
