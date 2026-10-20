"""Test suite for the Keras3 module."""

import keras


def gelu_activation(x: keras.KerasTensor, approximate: bool = False) -> keras.KerasTensor:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return keras.activations.gelu(x, approximate=approximate)
