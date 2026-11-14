"""Test suite for the Tensorflow module."""

import tensorflow as tf


class FlattenModel:
  """Test double `FlattenModel` implementing `call` for tensorflow tests."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance.

    Args:
        start_dim: Input `start_dim` parameter value (`int`).
    """
    super().__init__()
    self.flatten = tf.keras.layers.Flatten()

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Execute `flatten` to support call in tensorflow tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.flatten(x)
