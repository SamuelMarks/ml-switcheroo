"""Test suite for the Keras3 module."""

import keras


def concat_tensors(x: keras.KerasTensor, y: keras.KerasTensor, axis: int = -1) -> keras.KerasTensor:
  """Helper to concat tensors.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return keras.ops.concatenate([x, y], axis=axis)
