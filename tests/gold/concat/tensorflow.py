"""Test suite for the Tensorflow module."""

import tensorflow as tf


def concat_tensors(x: tf.Tensor, y: tf.Tensor, axis: int = -1) -> tf.Tensor:
  """Helper to concat tensors.

  Args:
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      axis: The axis (int) for concat tensors.

  Returns:
      Resulting `tf.Tensor` from concat tensors.
  """
  return tf.concat([x, y], axis=axis)
