"""Test suite for the Mlx module."""

import mlx.core as mx


def bmm_einsum(x: mx.array, y: mx.array) -> mx.array:
  """Execute `mx.matmul` to support bmm einsum in MLX tests.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.

  Returns:
      Resulting `mx.array` from bmm einsum.
  """
  return mx.matmul(x, y)
