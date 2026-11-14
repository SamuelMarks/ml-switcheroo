"""Apple MLX golden reference for data-dependent conditional control flow."""

import mlx.core as mx


def conditional_op(pred: mx.array, x: mx.array) -> mx.array:
  """Select ``x * 2`` when ``pred`` evaluates true and ``x + 2`` otherwise.

  Args:
      pred: Scalar boolean or numeric predicate array.
      x: Input MLX array operand.

  Returns:
      Elementwise or scalar-selected MLX array output.
  """
  res: mx.array = mx.where(pred, x * 2, x + 2)
  return res
