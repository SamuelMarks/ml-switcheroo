"""Module docstring."""

import keras


class ResidualBlock(keras.Model):
  """Class docstring."""

  def __init__(self, channels: int):
    """Function docstring."""
    super().__init__()
    self.conv1 = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn1 = keras.layers.BatchNormalization()
    self.conv2 = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn2 = keras.layers.BatchNormalization()

  def call(self, x, training=None):
    """Function docstring."""
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = keras.activations.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Keras supports raw + operator or keras.layers.Add
    out = out + residual
    # </SWITCHEROO_FAILED_TO_TRANS>
    out = keras.activations.relu(out)
    return out
