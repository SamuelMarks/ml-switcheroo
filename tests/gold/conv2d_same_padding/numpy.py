"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class SameConvModel:
  """Test helper class SameConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the SameConvModel instance.

    Args:
        in_channels: Number of input convolution channels.
        out_channels: Number of output convolution channels.
        kernel_size: Convolution or pooling spatial kernel size.
    """
    pass

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    return x
