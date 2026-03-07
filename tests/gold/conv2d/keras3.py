"""Module docstring."""

import keras


class ConvModel(keras.Model):
  """Class docstring."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Function docstring."""
    super().__init__()
    # Keras uses NHWC by default
    self.conv = keras.layers.Conv2D(out_channels, kernel_size, input_shape=(None, None, in_channels))

  def call(self, x):
    """Function docstring."""
    return self.conv(x)
