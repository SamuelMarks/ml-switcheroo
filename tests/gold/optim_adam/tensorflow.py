"""Module docstring."""

import tensorflow as tf


def setup_adam(model: tf.keras.Model, lr: float = 0.001):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  return tf.keras.optimizers.Adam(learning_rate=lr)
  # </SWITCHEROO_FAILED_TO_TRANS>
