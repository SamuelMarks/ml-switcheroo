"""Test suite for the Tensorflow module."""

import tensorflow as tf


class ConvModel:
  """Test helper class ConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the ConvModel instance.

    Args:
        in_channels: Number of input convolution channels.
        out_channels: Number of output convolution channels.
        kernel_size: Convolution or pooling spatial kernel size.
    """
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size, input_shape=(None, None, in_channels))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `tf.Tensor` from call.
    """
    return self.conv(x)
