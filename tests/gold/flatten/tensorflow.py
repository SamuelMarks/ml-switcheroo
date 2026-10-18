"""Test suite for the Tensorflow module."""

import tensorflow as tf


class FlattenModel:
  """Test helper class FlattenModel."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance.

    Args:
        start_dim: Test argument or fixture.
    """
    super().__init__()
    self.flatten = tf.keras.layers.Flatten()

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.flatten(x)
