"""Test suite for the Mlx module."""

import mlx.core as mx


def concat_tensors(x: mx.array, y: mx.array, axis: int = -1) -> mx.array:
  """Helper to concat tensors."""
  return mx.concatenate([x, y], axis=axis)
