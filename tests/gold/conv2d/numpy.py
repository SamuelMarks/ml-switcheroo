"""Test suite for the Numpy module."""

import typing

import numpy as np


class ConvModel:
  """Test helper class ConvModel."""

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
    """Initializes the ConvModel instance.

    Args:
        in_channels: Test argument or fixture.
        out_channels: Test argument or fixture.
        kernel_size: Test argument or fixture.
    """
    self.weight = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) / np.sqrt(
      in_channels * kernel_size * kernel_size
    )
    self.bias = np.zeros(out_channels)

  def __call__(self, x: np.ndarray[typing.Any, typing.Any]) -> np.ndarray[typing.Any, typing.Any]:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    (batch_size, height, width, _) = x.shape
    (kernel_h, kernel_w, _, out_c) = self.weight.shape
    out_h = height - kernel_h + 1
    out_w = width - kernel_w + 1
    out = np.zeros((batch_size, out_h, out_w, out_c))
    for i in range(out_h):
      for j in range(out_w):
        region = x[:, i : i + kernel_h, j : j + kernel_w, :]
        out[:, i, j, :] = np.tensordot(region, self.weight, axes=([1, 2, 3], [0, 1, 2])) + self.bias
    return out
