"""Test suite for the Tensorflow module."""

import tensorflow as tf


def gelu_activation(x: tf.Tensor, approximate: bool = False) -> tf.Tensor:
  """Helper to gelu activation.

  Args:
      x: Primary input tensor or array operand.
      approximate: The approximate (bool) for gelu activation.

  Returns:
      Resulting `tf.Tensor` from gelu activation.
  """
  return tf.nn.gelu(x, approximate=approximate)
