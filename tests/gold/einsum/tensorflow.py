"""Test suite for the Tensorflow module."""

import tensorflow as tf


def bmm_einsum(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  """Helper to bmm einsum."""
  return tf.einsum("bik,bkj->bij", x, y)
