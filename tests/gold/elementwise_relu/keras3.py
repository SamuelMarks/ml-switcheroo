"""Test suite for the Keras3 module."""

import keras


def relu_activation(x: keras.KerasTensor) -> keras.KerasTensor:
  """Execute `relu` to support relu activation in keras3 tests.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `keras.KerasTensor` from relu activation.
  """
  return keras.activations.relu(x)
