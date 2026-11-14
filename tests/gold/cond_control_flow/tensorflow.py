"""Test suite for the Tensorflow module."""

import tensorflow as tf


def conditional_op(pred: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
  """Execute `tf.cond` to support conditional operation in tensorflow tests.

  Args:
      pred: Input `pred` parameter value (`tf.Tensor`).
      x: Primary input tensor or array operand.

  Returns:
      Resulting `tf.Tensor` from conditional op.
  """
  return tf.cond(pred, lambda: x * 2, lambda: x + 2)
