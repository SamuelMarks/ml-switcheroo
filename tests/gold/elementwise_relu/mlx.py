"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


def relu_activation(x: mx.array) -> mx.array:
  """Helper to relu activation.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `mx.array` from relu activation.
  """
  return nn.relu(x)
