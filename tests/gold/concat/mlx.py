"""Test suite for the Mlx module."""

import mlx.core as mx


def concat_tensors(x: mx.array, y: mx.array, axis: int = -1) -> mx.array:
  """Helper to concat tensors.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      axis: The axis (int) for concat tensors.

  Returns:
      Resulting `mx.array` from concat tensors.
  """
  return mx.concatenate([x, y], axis=axis)
