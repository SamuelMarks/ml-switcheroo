"""Test suite for the Mlx module."""

import mlx.core as mx


def concat_tensors(x: mx.array, y: mx.array, axis: int = -1) -> mx.array:
  """Helper to concat tensors.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return mx.concatenate([x, y], axis=axis)
