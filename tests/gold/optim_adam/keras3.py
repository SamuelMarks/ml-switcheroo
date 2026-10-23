"""Test suite for the Keras3 module."""

import keras


def setup_adam(model: keras.Model, lr: float = 0.001) -> keras.KerasTensor:
  """Helper to setup adam.

  Args:
      model: Neural network model instance under test.
      lr: The lr (float) for setup adam.

  Returns:
      Resulting `keras.KerasTensor` from setup adam.
  """
  return keras.optimizers.Adam(learning_rate=lr)
