"""NumPy golden reference for Group Normalization."""

import numpy as np
import numpy.typing as npt


class GNModel:
  """Pure NumPy Group Normalization layer over NCHW tensors."""

  def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
    """Initialize scale, shift, and group configuration.

    Args:
        num_groups: Number of channel groups.
        num_channels: Total number of channels ``C``.
        eps: Numerical stability epsilon added to variance.
    """
    self.num_groups = num_groups
    self.num_channels = num_channels
    self.eps = eps
    self.weight = np.ones((1, num_channels, 1, 1), dtype=np.float64)
    self.bias = np.zeros((1, num_channels, 1, 1), dtype=np.float64)

  def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize NCHW input ``x`` within each channel group.

    Args:
        x: Input NCHW floating-point array.

    Returns:
      Group-normalized NCHW array.
    """
    n, c, h, w = x.shape
    grouped = x.reshape(n, self.num_groups, c // self.num_groups, h, w)
    mean = grouped.mean(axis=(2, 3, 4), keepdims=True)
    var = grouped.var(axis=(2, 3, 4), keepdims=True)
    normed = ((grouped - mean) / np.sqrt(var + self.eps)).reshape(n, c, h, w)
    return np.asarray(normed * self.weight + self.bias, dtype=np.float64)
