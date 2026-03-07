"""Module docstring."""

import keras


class SameConvModel(keras.Model):
  """Class docstring."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Function docstring."""
    super().__init__()
    self.conv = keras.layers.Conv2D(out_channels, kernel_size, padding="same")

  def call(self, x):
    """Function docstring."""
    return self.conv(x)
