"""Test suite for the Keras3 module."""

import keras


def bmm_einsum(x: keras.KerasTensor, y: keras.KerasTensor) -> keras.KerasTensor:
  """Execute `einsum` to support bmm einsum in keras3 tests.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.

  Returns:
      Resulting `keras.KerasTensor` from bmm einsum.
  """
  return keras.ops.einsum("bik,bkj->bij", x, y)
