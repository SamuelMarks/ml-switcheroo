"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


def gelu_activation(x: mx.array, approximate: mx.array = "none") -> mx.array:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  if approximate == "tanh":
    return nn.gelu_approx(x)
  return nn.gelu(x)
