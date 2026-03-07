"""Module docstring."""

import mlx.core as mx


def bmm_einsum(x: mx.array, y: mx.array) -> mx.array:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # MLX does not support full einsum string parsing natively, relies on matmul
  return mx.matmul(x, y)
  # </SWITCHEROO_FAILED_TO_TRANS>
