"""Test suite for the Keras3 module."""

import keras


class SameConvModel:
  """Test helper class SameConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
    """Initializes the SameConvModel instance.

    Args:
        in_channels: Number of input convolution channels.
        out_channels: Number of output convolution channels.
        kernel_size: Convolution or pooling spatial kernel size.
    """
    super().__init__()
    self.conv = keras.layers.Conv2D(out_channels, kernel_size, padding="same")

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.conv(x)
