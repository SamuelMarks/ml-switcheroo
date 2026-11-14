"""Test suite for the Tensorflow module."""

import tensorflow as tf


class GAPModel:
  """Test double `GAPModel` implementing `call` for tensorflow tests."""

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Execute `reduce_mean` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return tf.math.reduce_mean(x, axis=(1, 2))
