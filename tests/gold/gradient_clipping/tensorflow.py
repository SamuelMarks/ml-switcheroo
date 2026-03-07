"""Module docstring."""

import tensorflow as tf


def get_clipped_optimizer(learning_rate: float, max_norm: float = 1.0):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # TF can clip inside the optimizer constructor
  return tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=max_norm)
  # </SWITCHEROO_FAILED_TO_TRANS>
