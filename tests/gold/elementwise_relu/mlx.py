"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


def relu_activation(x: mx.array) -> mx.array:
  """Helper to relu activation."""
  return nn.relu(x)
