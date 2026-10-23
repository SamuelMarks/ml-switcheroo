"""Test suite for the Tensorflow module."""

import tensorflow as tf


def split_tensor(x: tf.Tensor, split_size: int, axis: int = -1) -> tf.Tensor:
  """Splits tensor.

  Args:
      x: Primary input tensor or array operand.
      split_size: The split size (int) for split tensor.
      axis: The axis (int) for split tensor.

  Returns:
      Resulting `tf.Tensor` from split tensor.
  """
  num_splits = x.shape[axis] // split_size
  return tf.split(x, num_splits, axis=axis)
