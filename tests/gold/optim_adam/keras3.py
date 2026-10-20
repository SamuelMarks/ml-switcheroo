"""Test suite for the Keras3 module."""

import keras


def setup_adam(model: keras.Model, lr: float = 0.001) -> keras.KerasTensor:
  """Helper to setup adam.

  Args:
      model: Test argument or fixture.
      lr: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return keras.optimizers.Adam(learning_rate=lr)
