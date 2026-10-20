"""Test suite for the Tensorflow module."""

import tensorflow as tf


def setup_adam(model: tf.keras.Model, lr: float = 0.001) -> tf.Tensor:
  """Helper to setup adam.

  Args:
      model: Test argument or fixture.
      lr: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return tf.keras.optimizers.Adam(learning_rate=lr)
