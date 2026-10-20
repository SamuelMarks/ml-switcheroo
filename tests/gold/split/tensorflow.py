"""Test suite for the Tensorflow module."""

import tensorflow as tf


def split_tensor(x: tf.Tensor, split_size: int, axis: int = -1) -> tf.Tensor:
  """Splits tensor.

  Args:
      x: Test argument or fixture.
      split_size: Test argument or fixture.
      axis: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  num_splits = x.shape[axis] // split_size
  return tf.split(x, num_splits, axis=axis)
