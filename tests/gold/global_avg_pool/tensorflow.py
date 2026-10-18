"""Test suite for the Tensorflow module."""

import tensorflow as tf


class GAPModel:
  """Test helper class GAPModel."""

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return tf.math.reduce_mean(x, axis=(1, 2))
