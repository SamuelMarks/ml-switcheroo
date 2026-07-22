"""Test suite for the Keras3 module."""

import keras


class FlattenModel(keras.Model):
  """Test suite for the Flatten Model component."""

  def __init__(self, start_dim: int = 1):
    """Initializes the FlattenModel instance."""
    super().__init__()
    self.flatten = keras.layers.Flatten()

  def call(self, x):
    """Helper to call."""
    return self.flatten(x)
