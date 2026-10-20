"""Test suite for the Tensorflow module."""

import tensorflow as tf


def get_clipped_optimizer(learning_rate: float, max_norm: float = 1.0) -> tf.Tensor:
  """Gets clipped optimizer.

  Args:
      learning_rate: Test argument or fixture.
      max_norm: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=max_norm)
