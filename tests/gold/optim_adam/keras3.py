"""Module docstring."""

import keras


def setup_adam(model: keras.Model, lr: float = 0.001):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  return keras.optimizers.Adam(learning_rate=lr)
  # </SWITCHEROO_FAILED_TO_TRANS>
