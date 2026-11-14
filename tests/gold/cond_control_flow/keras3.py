"""Keras 3 golden reference for data-dependent conditional control flow."""

import keras


def conditional_op(pred: keras.KerasTensor, x: keras.KerasTensor) -> keras.KerasTensor:
  """Select ``x * 2`` when ``pred`` is truthy and ``x + 2`` otherwise.

  Args:
      pred: Scalar boolean or numeric predicate tensor.
      x: Input tensor operand.

  Returns:
      Result tensor selected by ``keras.ops.cond``.
  """
  res: keras.KerasTensor = keras.ops.cond(pred, lambda: x * 2, lambda: x + 2)
  return res
