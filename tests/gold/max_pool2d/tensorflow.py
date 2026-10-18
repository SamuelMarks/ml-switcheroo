"""Test suite for the Tensorflow module."""

import tensorflow as tf


class MaxPoolModel:
  """Test helper class MaxPoolModel."""

  def __init__(self, pool_size: int = 2, strides: int = 2):
    """Initializes the MaxPoolModel instance.

    Args:
        pool_size: Test argument or fixture.
        strides: Test argument or fixture.
    """
    super().__init__()
    self.pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.pool(x)
