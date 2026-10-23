"""Test suite for the Tensorflow module."""

import tensorflow as tf


def conditional_op(pred: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
  """Helper to conditional op.

  Args:
      pred: The pred (tf.Tensor) for conditional op.
      x: Primary input tensor or array operand.

  Returns:
      Resulting `tf.Tensor` from conditional op.
  """
  return tf.cond(pred, lambda: x * 2, lambda: x + 2)
