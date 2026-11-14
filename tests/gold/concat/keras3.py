"""Test suite for the Keras3 module."""

import keras


def concat_tensors(x: keras.KerasTensor, y: keras.KerasTensor, axis: int = -1) -> keras.KerasTensor:
  """Execute `concatenate` to support concat tensors in keras3 tests.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      axis: Input `axis` parameter value (`int`).

  Returns:
      Resulting `keras.KerasTensor` from concat tensors.
  """
  return keras.ops.concatenate([x, y], axis=axis)
