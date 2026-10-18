"""Test suite for the Mlx module."""

import mlx.core as mx


def bmm_einsum(x: mx.array, y: mx.array) -> mx.array:
  """Helper to bmm einsum.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return mx.matmul(x, y)
