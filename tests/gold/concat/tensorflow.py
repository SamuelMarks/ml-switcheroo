"""Test suite for the Tensorflow module."""

import tensorflow as tf


def concat_tensors(x: tf.Tensor, y: tf.Tensor, axis: int = -1) -> tf.Tensor:
  """Helper to concat tensors."""
  return tf.concat([x, y], axis=axis)
