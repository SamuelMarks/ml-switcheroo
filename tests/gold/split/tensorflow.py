"""Test suite for the Tensorflow module."""

import tensorflow as tf


def split_tensor(x: tf.Tensor, split_size: int, axis: int = -1):
  """Splits tensor."""
  num_splits = x.shape[axis] // split_size
  return tf.split(x, num_splits, axis=axis)
