"""Test suite for the Tensorflow module."""

import tensorflow as tf


class MaxPoolModel:
  """Test double `MaxPoolModel` implementing `call` for tensorflow tests."""

  def __init__(self, pool_size: int = 2, strides: int = 2):
    """Initializes the MaxPoolModel instance.

    Args:
        pool_size: Input `pool_size` parameter value (`int`).
        strides: Input `strides` parameter value (`int`).
    """
    super().__init__()
    self.pool = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Execute `pool` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.pool(x)
