"""Test suite for the Keras3 module."""

import keras


def gelu_activation(x: keras.KerasTensor, approximate: bool = False) -> keras.KerasTensor:
  """Helper to gelu activation.

  Args:
      x: Primary input tensor or array operand.
      approximate: The approximate (bool) for gelu activation.

  Returns:
      Resulting `keras.KerasTensor` from gelu activation.
  """
  return keras.activations.gelu(x, approximate=approximate)
