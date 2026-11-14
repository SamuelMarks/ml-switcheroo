"""Test suite for the Tensorflow module."""

import tensorflow as tf


def relu_activation(x: tf.Tensor) -> tf.Tensor:
  """Execute `relu` to support relu activation in tensorflow tests.

  Args:
      x: Primary input tensor or array operand.

  Returns:
      Resulting `tf.Tensor` from relu activation.
  """
  return tf.nn.relu(x)
