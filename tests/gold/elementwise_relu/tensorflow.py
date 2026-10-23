"""Test suite for the Tensorflow module."""

import tensorflow as tf


def relu_activation(x: tf.Tensor) -> tf.Tensor:
  """Helper to relu activation.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `tf.Tensor` from relu activation.
  """
  return tf.nn.relu(x)
