"""Test suite for the Mlx module."""

import mlx.core as mx


def concat_tensors(x: mx.array, y: mx.array, axis: int = -1) -> mx.array:
  """Execute `mx.concatenate` to support concat tensors in MLX tests.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      axis: Input `axis` parameter value (`int`).

  Returns:
      Resulting `mx.array` from concat tensors.
  """
  return mx.concatenate([x, y], axis=axis)
