"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


def gelu_activation(x: mx.array, approximate: str = "none") -> mx.array:
  """Helper to gelu activation."""
  if approximate == "tanh":
    return nn.gelu_approx(x)
  return nn.gelu(x)
