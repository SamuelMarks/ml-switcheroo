"""Test suite for the Tensorflow module."""

import tensorflow as tf


def bmm_einsum(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  """Helper to bmm einsum.

  Args:
      x: Test argument or fixture.
      y: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return tf.einsum("bik,bkj->bij", x, y)
