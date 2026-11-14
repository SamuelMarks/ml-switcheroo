"""Test suite for the Keras3 module."""

import keras


def split_tensor(x: keras.KerasTensor, split_size: int, axis: int = -1) -> keras.KerasTensor:
  """Splits tensor.

  Args:
      x: Primary input tensor or array operand.
      split_size: Input `split_size` parameter value (`int`).
      axis: Input `axis` parameter value (`int`).

  Returns:
      Resulting `keras.KerasTensor` from split tensor.
  """
  num_splits = keras.ops.shape(x)[axis] // split_size
  return keras.ops.split(x, num_splits, axis=axis)
