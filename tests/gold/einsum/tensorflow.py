"""Module docstring."""

import tensorflow as tf


def bmm_einsum(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  """Function docstring."""
  # TensorFlow
  return tf.einsum("bik,bkj->bij", x, y)
