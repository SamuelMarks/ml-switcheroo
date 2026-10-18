"""Test suite for the Tensorflow module."""

import tensorflow as tf


def relu_activation(x: tf.Tensor) -> tf.Tensor:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return tf.nn.relu(x)
