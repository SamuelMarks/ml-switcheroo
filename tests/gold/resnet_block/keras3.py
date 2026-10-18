"""Test suite for the Keras3 module."""

import typing

import keras


class ResidualBlock:
  """Test helper class ResidualBlock."""

  def __init__(self, channels: int) -> None:
    """Initializes the ResidualBlock instance.

    Args:
        channels: Test argument or fixture.
    """
    super().__init__()
    self.conv1: typing.Any = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn1: typing.Any = keras.layers.BatchNormalization()
    self.conv2: typing.Any = keras.layers.Conv2D(channels, kernel_size=3, padding="same")
    self.bn2: typing.Any = keras.layers.BatchNormalization()

  def call(self, x: typing.Any, training: typing.Optional[bool] = None) -> typing.Any:
    """Helper to call.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    residual: typing.Any = x
    out: typing.Any = self.conv1(x)
    out = self.bn1(out, training=training)
    out = keras.activations.relu(out)
    out = self.conv2(out)
    out = self.bn2(out, training=training)
    out = out + residual
    out = keras.activations.relu(out)
    return out
