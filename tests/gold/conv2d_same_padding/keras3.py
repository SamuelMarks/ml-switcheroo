"""Test suite for the Keras3 module."""

import keras


class SameConvModel(keras.Model):
  """Test suite for the Same Conv Model component."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the SameConvModel instance."""
    super().__init__()
    self.conv = keras.layers.Conv2D(out_channels, kernel_size, padding="same")

  def call(self, x):
    """Helper to call."""
    return self.conv(x)
