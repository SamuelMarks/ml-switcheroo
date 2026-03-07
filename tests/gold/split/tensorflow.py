"""Module docstring."""

import tensorflow as tf


def split_tensor(x: tf.Tensor, split_size: int, axis: int = -1):
  """Function docstring."""
  # TF tf.split splits into num_or_size_splits
  num_splits = x.shape[axis] // split_size
  return tf.split(x, num_splits, axis=axis)
