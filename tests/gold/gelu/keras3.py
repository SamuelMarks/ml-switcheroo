"""Test suite for the Keras3 module."""

import keras


def gelu_activation(x: keras.KerasTensor, approximate: bool = False) -> keras.KerasTensor:
  """Execute `gelu` to support gelu activation in keras3 tests.

  Args:
      x: Primary input tensor or array operand.
      approximate: Input `approximate` parameter value (`bool`).

  Returns:
      Resulting `keras.KerasTensor` from gelu activation.
  """
  return keras.activations.gelu(x, approximate=approximate)
