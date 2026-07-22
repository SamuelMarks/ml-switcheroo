"""Test suite for the Tensorflow module."""

import tensorflow as tf


def relu_activation(x: tf.Tensor) -> tf.Tensor:
  """Helper to relu activation."""
  return tf.nn.relu(x)
