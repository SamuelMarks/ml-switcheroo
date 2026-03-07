"""Module docstring."""

import numpy as np


class MaxPoolModel:
  """Class docstring."""

  def __init__(self, kernel_size: int = 2, stride: int = 2):
    """Function docstring."""
    self.kernel_size = kernel_size
    self.stride = stride

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Manual max pooling implementation is complex
    batch_size, height, width, channels = x.shape
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
    # </SWITCHEROO_FAILED_TO_TRANS>
