"""Test suite for the Keras3 module."""

import keras


class ResidualBlock(keras.Model):
  """Test suite for the Residual Block component."""

  def __init__(self, channels: int):
    """Initializes the ResidualBlock instance."""
    super().__init__()
    self.conv1 = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn1 = keras.layers.BatchNormalization()
    self.conv2 = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn2 = keras.layers.BatchNormalization()

  def call(self, x, training=None):
    """Helper to call."""
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = keras.activations.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    out = out + residual
    out = keras.activations.relu(out)
    return out
