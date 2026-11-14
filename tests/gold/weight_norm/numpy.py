"""NumPy golden reference for Weight-Normalized Linear projection."""

import numpy as np
import numpy.typing as npt


class WNModel:
  """Pure NumPy weight-normalized linear transformation."""

  def __init__(self, in_features: int, out_features: int) -> None:
    """Initialize direction matrix ``weight_v``, scale ``weight_g``, and ``bias``.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
    """
    self.weight_v = np.ones((in_features, out_features), dtype=np.float64)
    self.weight_g = np.ones((1, out_features), dtype=np.float64)
    self.bias = np.zeros((out_features,), dtype=np.float64)

  def __call__(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute ``x @ (weight_g * weight_v / ||weight_v||) + bias``.

    Args:
        x: Input floating-point array ``(..., in_features)``.

    Returns:
        Weight-normalized linear projection array.
    """
    norm = np.linalg.norm(self.weight_v, axis=0, keepdims=True) + 1e-12
    w = self.weight_v * (self.weight_g / norm)
    return np.asarray(x @ w + self.bias, dtype=np.float64)
