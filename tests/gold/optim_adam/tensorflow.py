"""Test suite for the Tensorflow module."""

import tensorflow as tf


def setup_adam(model: tf.keras.Model, lr: float = 0.001) -> tf.Tensor:
  """Helper to setup adam.

  Args:
      model: Neural network model instance under test.
      lr: The lr (float) for setup adam.

  Returns:
      Resulting `tf.Tensor` from setup adam.
  """
  return tf.keras.optimizers.Adam(learning_rate=lr)
