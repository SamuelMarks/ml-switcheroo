"""Module docstring."""

import mlx.core as mx


def concat_tensors(x: mx.array, y: mx.array, axis: int = -1) -> mx.array:
  """Function docstring."""
  # MLX uses 'axis'
  return mx.concatenate([x, y], axis=axis)
