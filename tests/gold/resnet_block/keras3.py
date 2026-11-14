"""Test suite for the Keras3 module."""

import keras


class ResidualBlock:
  """Test double `ResidualBlock` implementing `call` for keras3 tests."""

  def __init__(self, channels: int) -> None:
    """Initializes the ResidualBlock instance.

    Args:
        channels: Input `channels` parameter value (`int`).
    """
    super().__init__()
    self.conv1 = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn1 = keras.layers.BatchNormalization()
    self.conv2 = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn2 = keras.layers.BatchNormalization()

  def call(self, x: keras.KerasTensor, training: bool | None = None) -> keras.KerasTensor:
    """Execute `conv1` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.
        training: Input `training` parameter value (`bool | None`).

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
    return keras.activations.relu(out)
