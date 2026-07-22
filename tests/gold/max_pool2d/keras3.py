"""Test suite for the Keras3 module."""

import keras


class MaxPoolModel(keras.Model):
  """Test suite for the Max Pool Model component."""

  def __init__(self, pool_size: int = 2, strides: int = 2):
    """Initializes the MaxPoolModel instance."""
    super().__init__()
    self.pool = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

  def call(self, x):
    """Helper to call."""
    return self.pool(x)
