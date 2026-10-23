"""Test suite for the Tensorflow module."""

import tensorflow as tf


def get_clipped_optimizer(learning_rate: float, max_norm: float = 1.0) -> tf.Tensor:
  """Gets clipped optimizer.

  Args:
      learning_rate: The learning rate (float) for get clipped optimizer.
      max_norm: The max norm (float) for get clipped optimizer.

  Returns:
      Resulting `tf.Tensor` from get clipped optimizer.
  """
  return tf.keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=max_norm)
