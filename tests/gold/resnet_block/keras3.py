"""Test suite for the Keras3 module."""

import keras


class ResidualBlock:
  """Test helper class ResidualBlock."""

  def __init__(self, channels: int) -> None:
    """Initializes the ResidualBlock instance.

    Args:
        channels: The channels (int) for   init  .
    """
    super().__init__()
    self.conv1 = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn1 = keras.layers.BatchNormalization()
    self.conv2 = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn2 = keras.layers.BatchNormalization()

  def call(self, x: keras.KerasTensor, training: bool | None = None) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.
        training: The training (bool | None) for call.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    residual = x
    out = self.conv1(x)
    out = self.bn1(out, training=training)
    out = keras.activations.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    out = out + residual
    out = keras.activations.relu(out)
    return out
