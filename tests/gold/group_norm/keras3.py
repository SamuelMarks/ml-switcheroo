"""Test suite for the Keras3 module."""

import keras


class GNModel(keras.Model):
  """Test suite for the G N Model component."""

  def __init__(self, num_groups: int, num_channels: int):
    """Initializes the GNModel instance."""
    super().__init__()
    self.gn = keras.layers.GroupNormalization(groups=num_groups)

  def call(self, x):
    """Helper to call."""
    return self.gn(x)
