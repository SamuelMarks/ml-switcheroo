"""Test suite for the Keras3 module."""

import keras


def relu_activation(x: keras.KerasTensor) -> keras.KerasTensor:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return keras.activations.relu(x)
