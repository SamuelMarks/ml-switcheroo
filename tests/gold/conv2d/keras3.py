"""Test suite for the Keras3 module."""

import keras


class ConvModel:
  """Test helper class ConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
    """Initializes the ConvModel instance.

    Args:
        in_channels: Test argument or fixture.
        out_channels: Test argument or fixture.
        kernel_size: Test argument or fixture.
    """
    super().__init__()
    self.conv = keras.layers.Conv2D(out_channels, kernel_size, input_shape=(None, None, in_channels))

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.conv(x)
