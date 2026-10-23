"""Test suite for the Tensorflow module."""

import tensorflow as tf


def bmm_einsum(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  """Helper to bmm einsum.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.

  Returns:
      Resulting `tf.Tensor` from bmm einsum.
  """
  return tf.einsum("bik,bkj->bij", x, y)
