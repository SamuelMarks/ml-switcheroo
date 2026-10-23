"""Test suite for the Numpy module."""

import numpy as np
import numpy.typing


class MaxPoolModel:
  """Test helper class MaxPoolModel."""

  def __init__(self, kernel_size: int = 2, stride: int = 2):
    """Initializes the MaxPoolModel instance.

    Args:
        kernel_size: Convolution or pooling spatial kernel size.
        stride: Convolution or pooling stride.
    """
    self.kernel_size = kernel_size
    self.stride = stride

  def __call__(self, x: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.float64]:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    (batch_size, height, width, channels) = x.shape
    out_h = (height - self.kernel_size) // self.stride + 1
    out_w = (width - self.kernel_size) // self.stride + 1
    out = np.zeros((batch_size, out_h, out_w, channels))
    for i in range(out_h):
      for j in range(out_w):
        h_start = i * self.stride
        w_start = j * self.stride
        out[:, i, j, :] = np.max(
          x[:, h_start : h_start + self.kernel_size, w_start : w_start + self.kernel_size, :], axis=(1, 2)
        )
    return out
