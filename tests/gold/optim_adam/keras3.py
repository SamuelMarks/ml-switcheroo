"""Test suite for the Keras3 module."""

import keras


def setup_adam(model: keras.Model, lr: float = 0.001) -> keras.KerasTensor:
  """Execute `Adam` to support setup adam in keras3 tests.

  Args:
      model: Neural network model instance under test.
      lr: Input `lr` parameter value (`float`).

  Returns:
      Resulting `keras.KerasTensor` from setup adam.
  """
  return keras.optimizers.Adam(learning_rate=lr)
