"""Test suite for the Keras3 module."""

import keras


def relu_activation(x: keras.KerasTensor) -> keras.KerasTensor:
  """Helper to relu activation.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `keras.KerasTensor` from relu activation.
  """
  return keras.activations.relu(x)
