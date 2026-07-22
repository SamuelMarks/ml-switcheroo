"""Test suite for the Mlx module."""

import mlx.core as mx


def bmm_einsum(x: mx.array, y: mx.array) -> mx.array:
  """Helper to bmm einsum."""
  return mx.matmul(x, y)
