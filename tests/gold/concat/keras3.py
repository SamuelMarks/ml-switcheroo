"""Test suite for the Keras3 module."""

import keras


def concat_tensors(x: keras.KerasTensor, y: keras.KerasTensor, axis: int = -1) -> keras.KerasTensor:
  """Helper to concat tensors.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      axis: The axis (int) for concat tensors.

  Returns:
      Resulting `keras.KerasTensor` from concat tensors.
  """
  return keras.ops.concatenate([x, y], axis=axis)
