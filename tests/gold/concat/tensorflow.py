"""Module docstring."""

import tensorflow as tf


def concat_tensors(x: tf.Tensor, y: tf.Tensor, axis: int = -1) -> tf.Tensor:
  """Function docstring."""
  # TensorFlow uses 'axis'
  return tf.concat([x, y], axis=axis)
