"""Test suite for the Keras3 module."""

import keras


def split_tensor(x: keras.KerasTensor, split_size: int, axis: int = -1) -> keras.KerasTensor:
  """Splits tensor.

  Args:
      x: Test argument or fixture.
      split_size: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  num_splits = keras.ops.shape(x)[axis] // split_size
  return keras.ops.split(x, num_splits, axis=axis)
