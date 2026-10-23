"""Test suite for the Tensorflow module."""

import tensorflow as tf


class GAPModel:
  """Test helper class GAPModel."""

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return tf.math.reduce_mean(x, axis=(1, 2))
