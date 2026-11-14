"""Test suite for the Tensorflow module."""

import tensorflow as tf


def concat_tensors(x: tf.Tensor, y: tf.Tensor, axis: int = -1) -> tf.Tensor:
  """Execute `tf.concat` to support concat tensors in tensorflow tests.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      axis: Input `axis` parameter value (`int`).

  Returns:
      Resulting `tf.Tensor` from concat tensors.
  """
  return tf.concat([x, y], axis=axis)
