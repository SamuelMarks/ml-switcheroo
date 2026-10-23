"""Test suite for the Tensorflow module."""

import tensorflow as tf


class FlattenModel:
  """Test helper class FlattenModel."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance.

    Args:
        start_dim: The start dim (int) for   init  .
    """
    super().__init__()
    self.flatten = tf.keras.layers.Flatten()

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.flatten(x)
