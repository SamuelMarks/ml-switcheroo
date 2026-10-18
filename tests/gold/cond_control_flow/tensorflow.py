"""Test suite for the Tensorflow module."""

import tensorflow as tf


def conditional_op(pred: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
  """Helper to conditional op.

  Args:
      pred: Test argument or fixture.
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return tf.cond(pred, lambda: x * 2, lambda: x + 2)
