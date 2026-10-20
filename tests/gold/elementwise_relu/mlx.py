"""Test suite for the Mlx module."""

import mlx.core as mx

import mlx.nn as nn


def relu_activation(x: mx.array) -> mx.array:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return nn.relu(x)
