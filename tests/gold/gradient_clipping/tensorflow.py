"""Test suite for the Tensorflow module."""

import tensorflow as tf


def get_clipped_optimizer(learning_rate: float, max_norm: float = 1.0):
  """Gets clipped optimizer."""
  return tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=max_norm)
